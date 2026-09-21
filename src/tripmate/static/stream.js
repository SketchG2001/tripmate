"use strict";
function createSSEParser(onEvent) {
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  function parse() {
    let boundary;
    while ((boundary = buffer.match(/\r?\n\r?\n/))) {
      const frame = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary[0].length);
      const data = frame.split(/\r?\n/).filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, "")).join("\n");
      if (data) onEvent(JSON.parse(data));
    }
    if (buffer.length > 1000000) throw new Error("Stream frame too large");
  }
  return {
    push(bytes) { buffer += decoder.decode(bytes, { stream: true }); parse(); },
    finish() {
      buffer += decoder.decode();
      parse();
      if (buffer.trim()) throw new Error("Incomplete stream frame");
    },
  };
}
