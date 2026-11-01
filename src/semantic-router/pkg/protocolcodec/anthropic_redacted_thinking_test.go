package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const redactedThinkingRequest = `{"model":"m","max_tokens":2048,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"redacted_thinking","data":"EmwKAhgBEgy3bGl0dGxlLXNlY3JldA=="},{"type":"text","text":"x"}]},{"role":"user","content":"y"}]}`

const redactedThinkingResponse = `{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[{"type":"redacted_thinking","data":"EmwKAhgBEgy3bGl0dGxlLXNlY3JldA=="},{"type":"text","text":"hello"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":2}}`

// redactedThinkingBareRequest omits the thinking budget so that cross-format
// targets reach the redacted-content gate instead of failing earlier on an
// unrelated capability (openai.responses.v1 rejects reasoning_budget first).
const redactedThinkingBareRequest = `{"model":"m","max_tokens":2048,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"redacted_thinking","data":"EmwKAhgBEgy3bGl0dGxlLXNlY3JldA=="},{"type":"text","text":"x"}]}]}`

func TestAnthropicRedactedThinkingRoundTripsRequest(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		[]byte(redactedThinkingRequest), nil,
	)
	if err != nil {
		t.Fatalf("same-format redacted_thinking request was rejected: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(result.Body, &body); err != nil {
		t.Fatal(err)
	}
	if !jsonContainsRedactedData(t, body, "EmwKAhgBEgy3bGl0dGxlLXNlY3JldA==") {
		t.Fatalf("redacted_thinking data was not preserved: %s", result.Body)
	}
}

func TestAnthropicRedactedThinkingRoundTripsResponse(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		[]byte(redactedThinkingResponse), nil,
	)
	if err != nil {
		t.Fatalf("same-format redacted_thinking response was rejected: %v", err)
	}
	if !strings.Contains(string(result.Body), `"type":"redacted_thinking"`) ||
		!strings.Contains(string(result.Body), `"data":"EmwKAhgBEgy3bGl0dGxlLXNlY3JldA=="`) {
		t.Fatalf("redacted_thinking block was not preserved: %s", result.Body)
	}
}

func TestAnthropicRedactedThinkingRoundTripsStream(t *testing.T) {
	payload := []byte(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"EmwKAhgBEgy3bGl0dGxlLXNlY3JldA==\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, err := stream.Push(payload)
	if err != nil {
		t.Fatalf("same-format redacted_thinking stream was rejected: %v", err)
	}
	finalFrames, finalEvents, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	_ = events
	_ = finalEvents
	joined := string(bytes.Join(append(append([][]byte(nil), frames...), finalFrames...), nil))
	joined = strings.ReplaceAll(joined, "\n", "")
	if !strings.Contains(joined, "redacted_thinking") || !strings.Contains(joined, "EmwKAhgBEgy3bGl0dGxlLXNlY3JldA==") {
		t.Fatalf("stream redacted_thinking block was not preserved: %s", joined)
	}
}

func TestAnthropicRedactedThinkingRejectsCrossFormat(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run("request/"+string(target), func(t *testing.T) {
			_, err := NewBuiltinEngine().TranslateRequest(
				llmprotocol.AnthropicMessagesV1, target, []byte(redactedThinkingBareRequest), nil,
			)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "redacted_reasoning")
		})
		t.Run("response/"+string(target), func(t *testing.T) {
			_, err := NewBuiltinEngine().TranslateResponse(
				llmprotocol.AnthropicMessagesV1, target, []byte(redactedThinkingResponse), nil,
			)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "redacted_reasoning")
		})
	}
}

func jsonContainsRedactedData(t *testing.T, value any, want string) bool {
	t.Helper()
	switch typed := value.(type) {
	case map[string]any:
		if data, ok := typed["data"].(string); ok && data == want {
			if blockType, ok := typed["type"].(string); ok && blockType == "redacted_thinking" {
				return true
			}
		}
		for _, child := range typed {
			if jsonContainsRedactedData(t, child, want) {
				return true
			}
		}
	case []any:
		for _, child := range typed {
			if jsonContainsRedactedData(t, child, want) {
				return true
			}
		}
	}
	return false
}
