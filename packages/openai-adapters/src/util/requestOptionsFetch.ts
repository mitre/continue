import { RequestOptions } from "@continuedev/config-types";
import { Readable } from "node:stream";

import { customFetch } from "../util.js";
import {
  nativeFetch,
  nativeHeaders,
  nativeRequest,
  nativeResponse,
  withNativeFetch,
} from "./nativeFetch.js";

/** Statuses the WHATWG Response constructor rejects a non-null body for. */
const NULL_BODY_STATUSES = new Set([101, 204, 205, 304]);

/**
 * True when requestOptions carry settings only customFetch can honor —
 * proxying and TLS configuration. Headers and timeout are excluded on
 * purpose: those already reach the @google/genai SDK via httpOptions.
 */
export function hasProxyOrTlsOptions(
  requestOptions: RequestOptions | undefined,
): boolean {
  return (
    !!requestOptions &&
    (requestOptions.proxy !== undefined ||
      requestOptions.verifySsl !== undefined ||
      requestOptions.caBundlePath !== undefined ||
      requestOptions.clientCertificate !== undefined)
  );
}

/**
 * Convert a node-fetch Response (Node Readable body, no getReader) into a
 * native WHATWG Response the @google/genai SDK can stream from. Without this
 * adaptation the SDK fails with "getReader is not a function" — the exact
 * pollution problem documented in nativeFetch.ts.
 */
export function adaptToNativeResponse(response: {
  status: number;
  statusText: string;
  headers: Iterable<[string, string]>;
  body: Readable | ReadableStream<Uint8Array> | null;
}): Response {
  const headers = new nativeHeaders([...response.headers]);

  let body: BodyInit | null = null;
  if (response.body !== null && !NULL_BODY_STATUSES.has(response.status)) {
    body =
      response.body instanceof Readable
        ? // Node's stream/web ReadableStream and the DOM lib type describe the
          // same runtime object; the cast bridges the two type declarations.
          (Readable.toWeb(response.body) as ReadableStream)
        : response.body;
  }

  return new nativeResponse(body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

/**
 * Run `fn` with globalThis.fetch honoring the given requestOptions.
 *
 * - No proxy/TLS options: delegates to withNativeFetch — byte-identical to
 *   the previous behavior for every existing config.
 * - Proxy/TLS options present: swaps globalThis.fetch for the duration of
 *   `fn` with a fetch that routes through customFetch(requestOptions)
 *   (proxy, CA bundles, client certs, verifySsl) and adapts its node-fetch
 *   Response to a native one so SDK streaming keeps working. The previous
 *   globals are restored in a finally, including when `fn` throws.
 */
export async function withRequestOptionsFetch<T>(
  requestOptions: RequestOptions | undefined,
  fn: () => Promise<T>,
): Promise<T> {
  if (!hasProxyOrTlsOptions(requestOptions)) {
    return withNativeFetch(fn);
  }

  const optionsFetch = customFetch(requestOptions);
  const wrappedFetch: typeof globalThis.fetch = async (input, init) => {
    const response = await optionsFetch(input, init);
    return adaptToNativeResponse(response);
  };

  const originalFetch = globalThis.fetch;
  const originalResponse = globalThis.Response;
  const originalRequest = globalThis.Request;
  const originalHeaders = globalThis.Headers;

  try {
    globalThis.fetch = wrappedFetch;
    globalThis.Response = nativeResponse;
    globalThis.Request = nativeRequest;
    globalThis.Headers = nativeHeaders;

    return await fn();
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.Response = originalResponse;
    globalThis.Request = originalRequest;
    globalThis.Headers = originalHeaders;
  }
}
