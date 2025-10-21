package gogent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// Client represents an LLM client that can generate responses
type LLMClient struct {
	openaiClient *openai.Client
	config       *Config
}

// NewClient creates a new LLM client with the given configuration
func NewClient(cfg *Config) *LLMClient {
	openaiClient := openai.NewClient(cfg.OpenAI.APIKey)
	return &LLMClient{
		openaiClient: openaiClient,
		config:       cfg,
	}
}

// shouldAddContext determines if we should automatically add context to the thread
func (c *LLMClient) shouldAddContext(thread *Thread) bool {
	// Check if there are any user messages in the thread
	hasUserMessages := false
	for _, msg := range thread.Messages {
		if msg.Role == "user" {
			hasUserMessages = true
			break
		}
	}

	// If there are user messages, we should add context
	return hasUserMessages
}

// GenerateResponse generates a response from the LLM based on the thread messages
func (c *LLMClient) GenerateResponse(ctx context.Context, thread *Thread, tools []Tool) (*Message, error) {
	if thread == nil {
		return nil, errors.New("thread cannot be nil")
	}

	//// Check if we need to automatically add context
	//if c.shouldAddContext(thread) {
	//	c.addContextToThread(ctx, thread, tools)
	//}

	//// Ограничиваем количество сообщений для предотвращения ошибки 429 Too Many Requests
	//maxHistoryMessages := c.config.MaxHistoryMessages
	//if maxHistoryMessages <= 0 {
	//	maxHistoryMessages = 15 // значение по умолчанию
	//}
	//trimmedMessages := thread.Messages
	//if len(trimmedMessages) > maxHistoryMessages {
	//	trimmedMessages = trimmedMessages[len(trimmedMessages)-maxHistoryMessages:]
	//	log.Printf("INFO: Trimmed thread messages from %d to %d to avoid token limit", len(thread.Messages), len(trimmedMessages))
	//}

	// Include all message types: system, user, assistant, and tool messages
	messages := make([]openai.ChatCompletionMessage, 0, 5) // +1 for system message

	//	// Add system message with instructions to use tools
	//	systemMessage := openai.ChatCompletionMessage{
	//		Role: "system",
	//		Content: `
	//You are a 999.md website assistant. You perform the functions of technical support and assistant
	//
	//You have access to the knowledge base with information about the rules, limits, prices and processes on the site.
	//
	//IMPORTANT: Always use the tools ("search_knowledge" and "search_rules") of the knowledge base before answering user questions!
	//
	//GENERAL RULES:
	//
	//IMPORTANT: Always answer in the same language in which the user asked the question (Russian → Russian, Romanian → Romanian, English → English).
	//
	//Communicate politely, friendly, using simple and understandable language.
	//
	//IMPORTANT: Do not add phrases similar to "There is no specific information in the 999.md knowledge base.." in response to the user, even if an exact indication of the problem was not found in the knowledge base.
	//
	//VERY IMPORTANT: DO NOT suggest specific steps/recommendations/actions not related to the user's question (for example: 'If you want, I can help with...' ; 'If you want, I can help with a detailed description of the steps...' ; 'If you want, I can describe each step in detail. Do you want?' ; 'I can help with detailed instructions. Do you want?' ; 'If you want, I can help with connecting a tariff or answer additional questions.' ; 'If necessary, I can help with confirming the number or explain how to correctly indicate contact information in the ad.' )
	//
	//If exact information is not found, make an educated guess or suggest contacting support by phone at +373 (22) 888 002 or by email at info@999.md, but never say "no information".
	//
	//USING THE TOOL:
	//
	//1. FOR ALL REQUESTS, you MUST first use the account data tool "get_user_info" so that you can respond to the user by addressing him by name.
	//
	//VERY IMPORTANT: pass the full user query to the "search_rules" and "search_knowledge" tools!!!
	//
	//2. FOR ALL REQUESTS, you MUST use the "search_rules" tool to get a list of rules with which you can answer the user's question
	//
	//3. FOR ALL REQUESTS, you MUST use the "search_knowledge" tool to get a list of knowledge, hints, and instructions with which you can answer the user's question
	//
	//4. Even if the "search_rules" tool finds suitable rules, you must use the "search_knowledge" tool to get knowledge
	//
	//5. use get_category_tree / get_category_by_id – to get information by category and subcategories
	//
	//6. Use get_user_chat_history to get previous messages in a conversation.
	//IMPORTANT: If the user's question consists of one word, for example (yes, okay, let's) or has a semantic coloring implying consent to the action that was offered to him - be sure to get previous messages and build an answer based on the dialogue!
	//
	//IMPORTANT:
	//Never answer questions about rules, restrictions, prices or processes without first checking the knowledge base using the "search_knowledge" and "search_rules" tools.
	//
	//Always request information from the available tools before forming an answer.
	//
	//VERY IMPORTANT: The user speaks Russian. You MUST answer ONLY in Russian.
	//
	//VERY IMPORTANT: Even if the user can understand other languages, answer ONLY in Russian.
	//
	//VERY IMPORTANT: Never switch to Romanian or English. Answer ONLY in Russian.
	//`,
	//	}
	//	messages = append(messages, systemMessage)

	for _, msg := range thread.Messages {
		// Sanitize missing role
		if msg.Role == "" {
			log.Printf("WARNING: Empty role in message; defaulting to 'user'")
			msg.Role = "user"
		}
		if msg.Role == "user" {
			// If MultiContent is present (type/text/image_url), send as MultiContent; otherwise send Content
			hasMC := msg.MultiContent.Type != "" || msg.MultiContent.Text != "" || (msg.MultiContent.ImageURL != nil && msg.MultiContent.ImageURL.URL != "")
			if hasMC {
				// Ensure type for image_url
				if msg.MultiContent.Type == "" && msg.MultiContent.ImageURL != nil && msg.MultiContent.ImageURL.URL != "" {
					msg.MultiContent.Type = openai.ChatMessagePartTypeImageURL
				}
				// Build parts: include text content as a separate part (if present), then the image/text part
				parts := make([]openai.ChatMessagePart, 0, 2)
				if strings.TrimSpace(msg.Content) != "" {
					parts = append(parts, openai.ChatMessagePart{Type: openai.ChatMessagePartTypeText, Text: msg.Content})
				}
				// Append the provided multiContent part (image or text)
				parts = append(parts, msg.MultiContent)
				openaiMsgUser := openai.ChatCompletionMessage{
					Role:         msg.Role,
					MultiContent: parts,
				}
				messages = append(messages, openaiMsgUser)
				continue
			}
			openaiMsgUser := openai.ChatCompletionMessage{
				Role:    msg.Role,
				Content: fmt.Sprintf("%v", msg.Content),
			}
			messages = append(messages, openaiMsgUser)
			continue
		}
		openaiMsg := openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: fmt.Sprintf("%v", msg.Content),
		}

		// For tool messages, include the tool_call_id
		if msg.Role == "tool" && msg.ToolCallID != "" {
			openaiMsg.ToolCallID = msg.ToolCallID
		}

		// For assistant messages with tool calls, include the tool calls
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			toolCalls := make([]openai.ToolCall, 0, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				// Convert arguments map to JSON string
				argsJSON, err := json.Marshal(tc.Arguments)
				if err != nil {
					return nil, err
				}

				toolCalls = append(toolCalls, openai.ToolCall{
					ID:   tc.ID,
					Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{
						Name:      tc.ToolName,
						Arguments: string(argsJSON),
					},
				})
			}
			openaiMsg.ToolCalls = toolCalls
		}

		messages = append(messages, openaiMsg)
	}

	// Convert agent tools to OpenAI tools
	openaiTools := make([]openai.Tool, 0, len(tools))
	for _, tool := range tools {
		// Convert parameters to JSON string for OpenAI API
		paramJSON, err := json.Marshal(tool.Parameters)
		if err != nil {
			return nil, err
		}

		openaiTools = append(openaiTools, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  json.RawMessage(paramJSON),
			},
		})
	}
	//m := openai.GPT4Dot1Mini

	//// Проверяем количество токенов и обрезаем историю, если необходимо
	//messagesTrimmed, err := TrimMessagesToFitTokenLimit(messages, c.config.OpenAIModel)
	//if err != nil {
	//	log.Printf("WARNING: Ошибка при оптимизации токенов: %v", err)
	//	// Аварийное сокращение сообщений (сохраняем системное и последние сообщения)
	//	if len(messages) > 5 {
	//		// Сохраняем системное сообщение и последние сообщения (удаляем старые)
	//		systemMsg := messages[0]
	//		// Определяем, сколько последних сообщений сохранить (4)
	//		keepCount := 4
	//		if len(messages)-1 < keepCount {
	//			keepCount = len(messages) - 1
	//		}
	//		// Берем последние keepCount сообщений (самые новые)
	//		lastMsgs := messages[len(messages)-keepCount:]
	//		messagesTrimmed = append([]openai.ChatCompletionMessage{systemMsg}, lastMsgs...)
	//		log.Printf("WARNING: Выполнено аварийное сокращение сообщений с %d до %d (удалены старые сообщения)", len(messages), len(messagesTrimmed))
	//	} else {
	//		messagesTrimmed = messages
	//	}
	//}

	//if len(messagesTrimmed) != len(messages) {
	//	log.Printf("INFO: Количество сообщений уменьшено с %d до %d из-за ограничения токенов", len(messages), len(messagesTrimmed))
	//}

	//// Получаем и устанавливаем максимальное количество токенов для модели
	//maxTokensForModel := GetMaxTokensForModel(c.config.OpenAIModel) - ReserveTokens
	//
	//// Проверка размера запроса (дополнительная проверка перед отправкой)
	//totalTokens, err := CountTokensInMessages(messagesTrimmed, c.config.OpenAIModel)
	//if err == nil && totalTokens > maxTokensForModel {
	//	log.Printf("WARNING: Запрос всё ещё содержит слишком много токенов (%d > %d) даже после оптимизации. Попытка дополнительного сокращения.",
	//		totalTokens, maxTokensForModel)
	//
	//	// Оставляем только системное сообщение и последние несколько сообщений (удаляем старые)
	//	if len(messagesTrimmed) > 4 {
	//		systemMsg := messagesTrimmed[0] // Сохраняем первое системное сообщение
	//		// Определяем, сколько последних сообщений сохранить (3)
	//		keepCount := 3
	//		if len(messagesTrimmed)-1 < keepCount {
	//			keepCount = len(messagesTrimmed) - 1
	//		}
	//		// Берём последние keepCount сообщений (самые новые)
	//		lastMsgs := messagesTrimmed[len(messagesTrimmed)-keepCount:]
	//		messagesTrimmed = append([]openai.ChatCompletionMessage{systemMsg}, lastMsgs...)
	//
	//		log.Printf("INFO: Выполнено экстренное сокращение с %d до %d сообщений (удалены старые сообщения)",
	//			len(messagesTrimmed)+keepCount, len(messagesTrimmed))
	//	}
	//}

	// Create the chat completion request
	request := openai.ChatCompletionRequest{
		Model:       c.config.OpenAI.VisionModelName,
		Messages:    messages,
		Tools:       openaiTools,
		Temperature: float32(0.0),
		//MaxTokens:   maxTokensForModel / 2, // Устанавливаем явное ограничение на максимальное количество токенов в ответе
	}

	// Call the OpenAI API
	response, err := c.openaiClient.CreateChatCompletion(ctx, request)
	if err != nil {
		if errMsg := err.Error(); strings.Contains(errMsg, "429") || strings.Contains(errMsg, "Too Many Requests") {
			log.Printf("ERROR: Received 429 Too Many Requests error despite token optimization: %v", err)
			return nil, fmt.Errorf("чат слишком большой для модели ИИ, пожалуйста, начните новый разговор: %w", err)
		}
		return nil, err
	}

	// Convert the OpenAI response to an agent message
	if len(response.Choices) == 0 {
		return nil, errors.New("no response from LLM")
	}

	choice := response.Choices[0]
	message := &Message{
		Role:    choice.Message.Role,
		Content: choice.Message.Content,
	}

	// Handle tool calls if present
	if len(choice.Message.ToolCalls) > 0 {
		toolCalls := make([]ToolCall, 0, len(choice.Message.ToolCalls))
		for _, tc := range choice.Message.ToolCalls {
			// Parse arguments string to map
			var arguments map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &arguments); err != nil {
				return nil, err
			}

			toolCalls = append(toolCalls, ToolCall{
				ID:        tc.ID,
				ToolName:  tc.Function.Name,
				Arguments: arguments,
			})
		}
		message.ToolCalls = toolCalls
	}

	return message, nil
}

// ExecuteToolCalls executes the tool calls in a message
func (c *LLMClient) ExecuteToolCalls(ctx context.Context, message *Message, tools []Tool) error {
	if message == nil {
		return errors.New("message cannot be nil")
	}

	// Create a map of tool names to tool functions for easy lookup
	toolMap := make(map[string]Tool)
	for _, tool := range tools {
		toolMap[tool.Name] = tool
	}

	// Execute each tool call
	for i, toolCall := range message.ToolCalls {
		tool, ok := toolMap[toolCall.ToolName]
		if !ok {
			log.Printf("ERROR: Tool not found: %s", toolCall.ToolName)
			// Instead of returning error, set error message in tool output
			message.ToolCalls[i].Output = fmt.Sprintf("Error: Tool '%s' not found. Please check available tools and try again.", toolCall.ToolName)
			continue
		}

		// Log the tool call with detailed information
		argsJSON, _ := json.Marshal(toolCall.Arguments)
		who := ""
		if toolCall.ToolName == "handoff" {
			who = fmt.Sprintf("(%s)", toolCall.Arguments["target_agent"])
		} else {
			who = ""
		}
		log.Printf("INFO-: Agent calling tool: %s %s with arguments: %s", toolCall.ToolName, who, string(argsJSON))

		// Execute the tool function
		result, err := tool.Function(ctx, toolCall.Arguments)
		if err != nil {
			log.Printf("ERROR: Tool %s execution failed: %v", toolCall.ToolName, err)
			// Instead of returning error, set error message in tool output
			// This allows LLM to handle the error gracefully
			errorMessage := fmt.Sprintf("Error executing tool '%s': %v. Please provide more specific information or try a different approach.", toolCall.ToolName, err)
			message.ToolCalls[i].Output = errorMessage
			continue
		}

		// Convert the result to a string
		var output string
		switch v := result.(type) {
		case string:
			output = v
		case []byte:
			output = string(v)
		default:
			// For other types, we could use JSON marshaling
			resultJSON, _ := json.Marshal(result)
			output = string(resultJSON)
		}

		// Log the tool result (truncated if too long)
		if len(output) > 500 {
			log.Printf("INFO: Tool %s returned result (truncated): %s...", toolCall.ToolName, output[:500])
		} else {
			log.Printf("INFO: Tool %s returned result: %s", toolCall.ToolName, output)
		}

		// Update the tool call with the output
		message.ToolCalls[i].Output = output
	}

	return nil
}
