// localtalk-fm — thin Foundation Models CLI for LocalTalk (macOS 27+)
//
// Protocol: newline-delimited JSON on stdin/stdout.
// Commands: status | reset | respond | quit
// Events: status | reset | delta | done | quit | error (ok:false)

import Foundation
import FoundationModels

@main
struct LocalTalkFoundationModels {
    static func main() async {
        var session: LanguageModelSession?
        let stdout = FileHandle.standardOutput

        func writeJSON(_ obj: [String: Any]) {
            guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []),
                  var line = String(data: data, encoding: .utf8)
            else { return }
            line.append("\n")
            if let out = line.data(using: .utf8) {
                stdout.write(out)
                // Ensure Python's readline sees events promptly during streaming.
                try? stdout.synchronize()
            }
        }

        func availabilityString(_ availability: SystemLanguageModel.Availability) -> String {
            switch availability {
            case .available:
                return "available"
            case .unavailable(let reason):
                return "unavailable(\(String(describing: reason)))"
            @unknown default:
                return "unknown"
            }
        }

        // Foundation.readLine() blocks until a full line or EOF — ideal for the pipe protocol.
        while let raw = readLine(strippingNewline: true) {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            guard let data = trimmed.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let cmd = json["cmd"] as? String
            else {
                writeJSON(["ok": false, "error": "invalid json request"])
                continue
            }

            switch cmd {
            case "status":
                let model = SystemLanguageModel.default
                writeJSON([
                    "ok": true,
                    "event": "status",
                    "available": model.isAvailable,
                    "availability": availabilityString(model.availability),
                    "provider": "apple.foundation_models",
                    "model": "SystemLanguageModel.default",
                    "macos": ProcessInfo.processInfo.operatingSystemVersionString,
                ])

            case "reset":
                let instructions = (json["instructions"] as? String) ?? ""
                guard SystemLanguageModel.default.isAvailable else {
                    writeJSON([
                        "ok": false,
                        "error": "SystemLanguageModel unavailable: \(availabilityString(SystemLanguageModel.default.availability))",
                    ])
                    continue
                }
                if instructions.isEmpty {
                    session = LanguageModelSession()
                } else {
                    session = LanguageModelSession(instructions: instructions)
                }
                writeJSON(["ok": true, "event": "reset"])

            case "respond":
                guard SystemLanguageModel.default.isAvailable else {
                    writeJSON([
                        "ok": false,
                        "error": "SystemLanguageModel unavailable: \(availabilityString(SystemLanguageModel.default.availability))",
                    ])
                    continue
                }
                guard let prompt = json["prompt"] as? String, !prompt.isEmpty else {
                    writeJSON(["ok": false, "error": "missing prompt"])
                    continue
                }
                let stream = (json["stream"] as? Bool) ?? true
                let temperature = json["temperature"] as? Double
                let maxTokens = json["max_tokens"] as? Int

                if session == nil {
                    let instructions = (json["instructions"] as? String) ?? ""
                    session = instructions.isEmpty
                        ? LanguageModelSession()
                        : LanguageModelSession(instructions: instructions)
                }
                guard let active = session else {
                    writeJSON(["ok": false, "error": "failed to create session"])
                    continue
                }

                let options: GenerationOptions
                if temperature != nil || maxTokens != nil {
                    options = GenerationOptions(
                        sampling: nil,
                        temperature: temperature,
                        maximumResponseTokens: maxTokens
                    )
                } else {
                    options = GenerationOptions()
                }

                do {
                    if stream {
                        let responseStream = active.streamResponse(to: prompt, options: options)
                        var last = ""
                        for try await snapshot in responseStream {
                            let content = snapshot.content
                            if content != last {
                                last = content
                                writeJSON(["ok": true, "event": "delta", "content": content])
                            }
                        }
                        writeJSON(["ok": true, "event": "done", "content": last])
                    } else {
                        let response = try await active.respond(to: prompt, options: options)
                        writeJSON(["ok": true, "event": "done", "content": response.content])
                    }
                } catch {
                    writeJSON(["ok": false, "error": String(describing: error)])
                }

            case "quit":
                writeJSON(["ok": true, "event": "quit"])
                return

            default:
                writeJSON(["ok": false, "error": "unknown cmd \(cmd)"])
            }
        }
    }
}
