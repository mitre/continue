import { execSync } from "node:child_process";
import fs from "node:fs";
import https from "node:https";
import { AddressInfo } from "node:net";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { GeminiApi } from "../apis/Gemini.js";

/**
 * TLS behavior through the REAL @google/genai SDK and the REAL
 * customFetch/fetchwithRequestOptions stack (no mocks): a local HTTPS stub
 * Gemini server with a self-signed certificate. Certificate generation
 * mirrors packages/fetch/src/fetch.e2e.test.ts.
 */

let tempDir: string;
let httpsServer: https.Server;
let httpsPort: number;
let certPath: string;

function sseChunk(text: string, finishReason?: string): string {
  const candidate: Record<string, unknown> = {
    content: { role: "model", parts: [{ text }] },
    index: 0,
  };
  if (finishReason) {
    candidate.finishReason = finishReason;
  }
  return `data: ${JSON.stringify({ candidates: [candidate] })}\r\n\r\n`;
}

/** Self-signed server certificate — same openssl recipe as packages/fetch. */
function generateCertificate(dir: string): {
  certPath: string;
  keyPath: string;
} {
  const cert = path.join(dir, "server.crt");
  const key = path.join(dir, "server.key");
  const conf = path.join(dir, "server.conf");
  fs.writeFileSync(
    conf,
    `
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = Test
L = Test
O = Continue TLS Test
CN = 127.0.0.1

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
IP.1 = 127.0.0.1
DNS.1 = localhost
`,
  );
  execSync(`openssl genrsa -out "${key}" 2048`, { stdio: "pipe" });
  execSync(
    `openssl req -new -x509 -key "${key}" -out "${cert}" -days 365 -config "${conf}" -extensions v3_req`,
    { stdio: "pipe" },
  );
  return { certPath: cert, keyPath: key };
}

beforeAll(async () => {
  tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "gemini-tls-test-"));
  const generated = generateCertificate(tempDir);
  certPath = generated.certPath;

  httpsServer = https.createServer(
    {
      cert: fs.readFileSync(generated.certPath),
      key: fs.readFileSync(generated.keyPath),
    },
    (_req, res) => {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.write(sseChunk("secure ", undefined));
      res.write(sseChunk("stream", "STOP"));
      res.end();
    },
  );
  await new Promise<void>((resolve) =>
    httpsServer.listen(0, "127.0.0.1", resolve),
  );
  httpsPort = (httpsServer.address() as AddressInfo).port;
});

afterAll(async () => {
  await new Promise((resolve) => httpsServer.close(resolve));
  fs.rmSync(tempDir, { recursive: true, force: true });
});

function makeApi(requestOptions: Record<string, unknown>): GeminiApi {
  return new GeminiApi({
    provider: "gemini",
    apiKey: "stub-key",
    apiBase: `https://127.0.0.1:${httpsPort}/v1beta/`,
    requestOptions,
  });
}

async function drainChat(api: GeminiApi): Promise<string> {
  let content = "";
  for await (const chunk of api.chatCompletionStream(
    {
      model: "gemini-2.5-flash",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    },
    new AbortController().signal,
  )) {
    content += chunk.choices[0]?.delta?.content ?? "";
  }
  return content;
}

describe("Gemini TLS through the real SDK (no mocks)", () => {
  it("rejects a self-signed server when verifySsl is on", async () => {
    // verifySsl: true engages the wrapper; the unknown self-signed cert must
    // fail verification — proving TLS is genuinely enforced on this path.
    await expect(drainChat(makeApi({ verifySsl: true }))).rejects.toThrow(
      /self-signed|self signed|unable to verify|certificate/i,
    );
  });

  it("accepts the server when its certificate is trusted via caBundlePath", async () => {
    const content = await drainChat(makeApi({ caBundlePath: certPath }));
    expect(content).toBe("secure stream");
  });

  it("accepts the server when verifySsl is explicitly disabled", async () => {
    const content = await drainChat(makeApi({ verifySsl: false }));
    expect(content).toBe("secure stream");
  });
});
