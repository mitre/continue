import http from "node:http";
import { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { GeminiApi } from "../apis/Gemini.js";

/**
 * End-to-end proof for the proxy path: the REAL @google/genai SDK, the REAL
 * customFetch/fetchwithRequestOptions stack (no mocks anywhere), a real local
 * HTTP forward proxy, and a stub Gemini server emitting SSE. Verifies the
 * node-fetch Response adaptation streams through the SDK's own parser.
 */

let geminiServer: http.Server;
let proxyServer: http.Server;
let geminiPort: number;
let proxyPort: number;

/** Requests the stub Gemini server actually received. */
const geminiRequests: {
  url: string;
  apiKey: string | undefined;
  customHeader: string | undefined;
}[] = [];
/** Number of requests that transited the proxy. */
let proxiedRequests = 0;

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

beforeAll(async () => {
  geminiServer = http.createServer((req, res) => {
    geminiRequests.push({
      url: req.url ?? "",
      apiKey: req.headers["x-goog-api-key"] as string | undefined,
      customHeader: req.headers["x-custom-gateway"] as string | undefined,
    });
    if ((req.url ?? "").includes("mbedContent")) {
      // embedContent / batchEmbedContents — Google's real response shape
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ embeddings: [{ values: [0.25, 0.75] }] }));
      return;
    }
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    res.write(sseChunk("Hello "));
    res.write(sseChunk("from the proxied stub", "STOP"));
    res.end();
  });
  await new Promise<void>((resolve) =>
    geminiServer.listen(0, "127.0.0.1", resolve),
  );
  geminiPort = (geminiServer.address() as AddressInfo).port;

  // Minimal HTTP forward proxy: receives absolute-URI requests, forwards
  // them, pipes the response back.
  proxyServer = http.createServer((req, res) => {
    proxiedRequests += 1;
    const target = new URL(req.url ?? "");
    const upstream = http.request(
      {
        hostname: target.hostname,
        port: target.port,
        path: target.pathname + target.search,
        method: req.method,
        headers: { ...req.headers, host: target.host },
      },
      (upstreamRes) => {
        res.writeHead(upstreamRes.statusCode ?? 502, upstreamRes.headers);
        upstreamRes.pipe(res);
      },
    );
    req.pipe(upstream);
  });
  await new Promise<void>((resolve) =>
    proxyServer.listen(0, "127.0.0.1", resolve),
  );
  proxyPort = (proxyServer.address() as AddressInfo).port;
});

afterAll(async () => {
  await new Promise((resolve) => geminiServer.close(resolve));
  await new Promise((resolve) => proxyServer.close(resolve));
});

describe("Gemini streaming through a real local proxy (no mocks)", () => {
  it("streams SSE chunks via the real SDK, customFetch, and proxy", async () => {
    const api = new GeminiApi({
      provider: "gemini",
      apiKey: "stub-key",
      apiBase: `http://127.0.0.1:${geminiPort}/v1beta/`,
      requestOptions: {
        proxy: `http://127.0.0.1:${proxyPort}`,
      },
    });

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

    expect(content).toBe("Hello from the proxied stub");
    expect(proxiedRequests).toBe(1);
    expect(geminiRequests).toHaveLength(1);
    expect(geminiRequests[0].apiKey).toBe("stub-key");
    expect(geminiRequests[0].url).toContain(":streamGenerateContent");
  });

  it("delivers custom requestOptions.headers to the wire", async () => {
    const before = geminiRequests.length;
    const api = new GeminiApi({
      provider: "gemini",
      apiKey: "stub-key",
      apiBase: `http://127.0.0.1:${geminiPort}/v1beta/`,
      requestOptions: {
        proxy: `http://127.0.0.1:${proxyPort}`,
        headers: { "x-custom-gateway": "gw-credential-123" },
      },
    });

    for await (const _chunk of api.chatCompletionStream(
      {
        model: "gemini-2.5-flash",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      },
      new AbortController().signal,
    )) {
      // drain
    }

    const observed = geminiRequests[before];
    expect(observed.customHeader).toBe("gw-credential-123");
  });

  it("routes embed through the same proxy and parses Google's real shape", async () => {
    const proxiedBefore = proxiedRequests;
    const requestsBefore = geminiRequests.length;
    const api = new GeminiApi({
      provider: "gemini",
      apiKey: "stub-key",
      apiBase: `http://127.0.0.1:${geminiPort}/v1beta/`,
      requestOptions: {
        proxy: `http://127.0.0.1:${proxyPort}`,
      },
    });

    const result = await api.embed({
      model: "gemini-embedding-001",
      input: ["hello"],
    });

    expect(proxiedRequests).toBe(proxiedBefore + 1);
    expect(geminiRequests[requestsBefore].url).toContain("mbedContent");
    expect(result.data[0].embedding).toEqual([0.25, 0.75]);
  });
});
