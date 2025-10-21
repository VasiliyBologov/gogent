package gogent

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/sashabaranov/go-openai"
)

//// addContextToThread automatically adds context to the thread using getUserChatHistoryTool
//func (c *LLMClient) addContextToThread(ctx context.Context, thread *Thread, tools []Tool) {
//	log.Printf("INFO: Adding context to thread automatically")

//// Find the getUserChatHistoryTool
//var getUserChatHistoryTool *Tool
//for i := range tools {
//	if tools[i].Name == "get_user_chat_history" {
//		getUserChatHistoryTool = &tools[i]
//		break
//	}
//}
//
//if getUserChatHistoryTool == nil {
//	log.Printf("WARNING: getUserChatHistoryTool not found")
//	return
//}

//// Get user ID from thread metadata
//userID, ok := thread.Metadata["user_id"].(string)
//if !ok || userID == "" {
//	log.Printf("WARNING: No user ID found in thread metadata")
//	return
//}

//log.Printf("INFO: Getting chat history for user: %s to add context", userID)

//// Call the tool to get chat history
//result, err := getUserChatHistoryTool.Function(ctx, map[string]interface{}{
//	"user_id": userID,
//	"limit":   5,
//})

//if err != nil {
//	log.Printf("WARNING: Failed to get chat history for context: %v", err)
//	return
//}

//// Convert result to string
//var historyContent string
//switch v := result.(type) {
//case string:
//	historyContent = v
//case []byte:
//	historyContent = string(v)
//default:
//	resultJSON, _ := json.Marshal(result)
//	historyContent = string(resultJSON)
//}

//// Add context as a system message if we got some history
//if historyContent != "" && historyContent != "[]" {
//	contextMessage := Message{
//		Role:    "system",
//		Content: "Previous conversation context:\n" + historyContent,
//	}
//
//	// Insert context message at the beginning of the thread
//	thread.Messages = append([]Message{contextMessage}, thread.Messages...)
//	log.Printf("INFO: Successfully added context to thread for user: %s", userID)
//} else {
//	log.Printf("INFO: No previous context found for user: %s", userID)
//}
//}

// Tool represents a function that an agent can use to interact with external systems
type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Function    func(ctx context.Context, params map[string]interface{}) (interface{}, error)
}

// Message represents a message in a conversation
type Message struct {
	Role         string                 `json:"role"` // "user", "assistant", "system", "tool"
	Content      string                 `json:"content"`
	MultiContent openai.ChatMessagePart `json:"multiContent"`
	ToolCalls    []ToolCall             `json:"tool_calls,omitempty"`
	ToolCallID   string                 `json:"tool_call_id,omitempty"` // Used for tool messages
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt    time.Time              `json:"created_at"`
}

// ToolCall represents a call to a tool by the agent
type ToolCall struct {
	ID        string                 `json:"id"`
	ToolName  string                 `json:"tool_name"`
	Arguments map[string]interface{} `json:"arguments"`
	Output    string                 `json:"output,omitempty"`
}

// Thread represents a conversation thread
type Thread struct {
	ID        string                 `json:"id"`
	Messages  []Message              `json:"messages"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
}

// RunStatus represents the status of a run
type RunStatus string

const (
	RunStatusQueued     RunStatus = "queued"
	RunStatusInProgress RunStatus = "in_progress"
	RunStatusCompleted  RunStatus = "completed"
	RunStatusFailed     RunStatus = "failed"
	RunStatusCancelled  RunStatus = "cancelled"
)

// Run represents an execution of an agent on a thread
type Run struct {
	ID        string                 `json:"id"`
	ThreadID  string                 `json:"thread_id"`
	Status    RunStatus              `json:"status"`
	Tools     []Tool                 `json:"tools"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
	StartedAt *time.Time             `json:"started_at,omitempty"`
	EndedAt   *time.Time             `json:"ended_at,omitempty"`
}

// AgentConfig holds configuration for an agent
type AgentConfig struct {
	Model       string                 `json:"model"`
	Tools       []Tool                 `json:"tools"`
	MaxSteps    int                    `json:"max_steps"`
	Temperature float64                `json:"temperature"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// Handoff contains configuration/state for agent-to-agent delegation (not a tool)
type Handoff struct {
	Enabled  bool     `json:"enabled"`
	Children []string `json:"children,omitempty"`
}

// Agent represents an OpenAI agent that can perform tasks
type Agent struct {
	Config       AgentConfig
	client       *LLMClient
	systemPrompt string
	description  string
	//repo         TODO:  make possible add repo for tools
	childAgents map[string]*Agent
	Handoff     Handoff
}

// NewAgent creates a new agent with the given configuration
func NewAgent(config AgentConfig) *Agent {
	return &Agent{
		Config: config,
	}
}

// SetClient sets the LLM client for the agent
func (a *Agent) SetClient(client *LLMClient) {
	a.client = client
}

// SetDescription sets the Description for the agent
func (a *Agent) SetDescription(s string) {
	a.description = s
}

// SetSystemPrompt sets the SystemPrompt for the agent
func (a *Agent) SetSystemPrompt(s string) {
	a.systemPrompt = s
}

//// SetRepository sets the repository for the agent
//func (a *Agent) SetRepository(repo structure_999.ServiceStructurePublicClient) {
//	a.repo = repo
//}

//// GetRepository returns the repository for the agent
//func (a *Agent) GetRepository() interface{} {
//	return a.repo
//}

// RegisterChild registers a child agent by name for handoffs
func (a *Agent) RegisterChild(name string, child *Agent) {
	if a.childAgents == nil {
		a.childAgents = make(map[string]*Agent)
	}
	if child == nil || name == "" {
		return
	}
	// Inherit repository and client if child doesn't have them
	//if child.repo == nil {
	//	child.repo = a.repo
	//}
	if child.client == nil {
		child.client = a.client
	}
	a.childAgents[name] = child

	// Maintain Handoff state (separate property)
	a.Handoff.Enabled = true
	// Ensure unique child name in Handoff.Children
	exists := false
	for _, n := range a.Handoff.Children {
		if n == name {
			exists = true
			break
		}
	}
	if !exists {
		a.Handoff.Children = append(a.Handoff.Children, name)
	}
}

// GetChild retrieves a registered child agent by name
func (a *Agent) GetChild(name string) (*Agent, bool) {
	if a.childAgents == nil {
		return nil, false
	}
	ag, ok := a.childAgents[name]
	return ag, ok
}

// getChildNames returns the names of registered child agents
func (a *Agent) getChildNames() []string {
	if a.childAgents == nil {
		return nil
	}
	names := make([]string, 0, len(a.childAgents))
	for name := range a.childAgents {
		names = append(names, name)
	}
	return names
}

// getToolsForRun returns the tools to expose to the LLM for this run.
// Handoff remains a separate Agent property (not part of Config.Tools),
// but when enabled with registered children we expose a built-in synthetic
// "handoff" tool to allow the LLM to perform delegation.
func (a *Agent) getToolsForRun(base []Tool) []Tool {
	tools := make([]Tool, len(base))
	copy(tools, base)
	if a.Handoff.Enabled && len(a.childAgents) > 0 {
		tools = append(tools, a.makeHandoffTool())
	}
	return tools
}

// makeHandoffTool constructs a special tool that allows the LLM to delegate to child agents
func (a *Agent) makeHandoffTool() Tool {
	// Build enum of available child agents
	enumVals := make([]interface{}, 0, len(a.childAgents))
	for _, name := range a.getChildNames() {
		enumVals = append(enumVals, name)
	}
	params := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"target_agent": map[string]interface{}{
				"type":        "string",
				"description": "Name of the child agent to delegate to",
				"enum":        enumVals,
			},
			"input": map[string]interface{}{
				"type":        "object",
				"description": "User input or task for the child agent",
				"properties": map[string]interface{}{
					"role": map[string]interface{}{
						"type":        "string",
						"description": "role of the user message. Only 'user' is supported",
					},
					"content": map[string]interface{}{
						"type":        "string",
						"description": "Text of the user message if is text message.",
					},
					"multiContent": map[string]interface{}{
						"type":        "object",
						"description": "Users message if is multiContent message.",
						"properties": map[string]interface{}{
							"type": map[string]interface{}{
								"type":        "string",
								"description": "Type of the message. Only 'text' or 'image_url' is supported",
							},
							"text": map[string]interface{}{
								"type":        "string",
								"description": "Text of the message if is text message.",
							},
							"image_url": map[string]interface{}{
								"type":        "object",
								"description": "If message is image_url.",
								"properties": map[string]interface{}{
									"url": map[string]interface{}{
										"type":        "string",
										"description": "URL of the image.",
									},
									"detail": map[string]interface{}{
										"type":        "string",
										"description": "Detalization of the image. Supported only high / low / auto",
									},
								},
							},
						},
					},
				},
			},
			"mode": map[string]interface{}{
				"type":        "string",
				"description": "Delegation mode. Only 'blocking' is supported and returns child's result to parent.",
				"enum":        []interface{}{"blocking"},
			},
			"metadata": map[string]interface{}{
				"type":        "object",
				"description": "Optional metadata to pass to the child thread",
			},
		},
		"required": []interface{}{"target_agent", "input"},
	}

	return Tool{
		Name:        "handoff",
		Description: "Delegate the task to a specialized child agent and return its result.",
		Parameters:  params,
		Function: func(ctx context.Context, p map[string]interface{}) (interface{}, error) {
			target, _ := p["target_agent"].(string)
			input, err := toMessage(p["input"])
			if err != nil {
				return nil, fmt.Errorf("invalid input for handoff: %w", err)
			}
			// Для корректной валидации прокладываем parent_thread в ctx
			var parentThread *Thread
			if tval := ctx.Value("current_thread"); tval != nil {
				if t, ok := tval.(*Thread); ok {
					parentThread = t
				}
			}
			ctxWithThread := ctx
			if parentThread != nil {
				ctxWithThread = context.WithValue(ctx, "parent_thread", parentThread)
			}
			return a.DelegateToChild(ctxWithThread, target, &input, toMap(p["metadata"]))
		},
	}
}

// DelegateToChild performs a programmatic handoff to a registered child agent and returns the child's final response.
func (a *Agent) DelegateToChild(ctx context.Context, target string, input *Message, metadata map[string]interface{}) (map[string]interface{}, error) {
	if target == "" || input == nil {
		return nil, errors.New("handoff requires target and input")
	}
	child, ok := a.GetChild(target)
	if !ok || child == nil {
		return nil, fmt.Errorf("child agent '%s' not found", target)
	}

	// === КОНТРОЛЬ КОПИИ ЮЗЕРСКОГО IMAGE-URL ===
	// Ищем последний image url пользователя из thread/контекста
	// (ищем в последних сообщениях текущего thread, если есть)
	var lastUserImageURL string
	if threadVal := ctx.Value("parent_thread"); threadVal != nil {
		if thread, ok := threadVal.(*Thread); ok {
			for i := len(thread.Messages) - 1; i >= 0; i-- {
				msg := thread.Messages[i]
				if msg.Role == "user" && msg.MultiContent.Type == "image_url" && msg.MultiContent.ImageURL != nil && msg.MultiContent.ImageURL.URL != "" {
					lastUserImageURL = msg.MultiContent.ImageURL.URL
					break
				}
			}
		}
	}

	// Извлекаем url из переданного input (парам для handoff)
	var handoffImageUrl string
	if input != nil && input.MultiContent.Type == "image_url" && input.MultiContent.ImageURL != nil {
		handoffImageUrl = input.MultiContent.ImageURL.URL

		// --- АВТОМАТИЧЕСКИ ДОБАВЛЯЕМ 'detail' "low", если отсутствует
		if input.MultiContent.ImageURL.Detail == "" {
			input.MultiContent.ImageURL.Detail = "low"
		}
	}
	// Также устанавливаем тип multiContent явно, если модель его случайно не проставила, но картинка есть
	if input != nil && input.MultiContent.Type == "" && input.MultiContent.ImageURL != nil && input.MultiContent.ImageURL.URL != "" {
		input.MultiContent.Type = "image_url"
	}
	// И role если отсутствует
	if input != nil && input.Role == "" {
		input.Role = "user"
	}

	// URL pass-through enforcement: always use the user's last image URL if available.
	if lastUserImageURL != "" {
		if handoffImageUrl == "" || lastUserImageURL != handoffImageUrl {
			log.Printf("WARNING: Overriding handoff image_url to user's URL. User: %s, Handoff: %s", lastUserImageURL, handoffImageUrl)
			if input.MultiContent.ImageURL == nil {
				input.MultiContent.ImageURL = &openai.ChatMessageImageURL{}
			}
			input.MultiContent.ImageURL.URL = lastUserImageURL
			if input.MultiContent.Type == "" {
				input.MultiContent.Type = openai.ChatMessagePartTypeImageURL
			}
			if input.MultiContent.ImageURL.Detail == "" {
				input.MultiContent.ImageURL.Detail = openai.ImageURLDetailLow
			}
			handoffImageUrl = lastUserImageURL
		}
	}

	// Extract user ID from context if available
	var userID string
	if v := ctx.Value("user_id"); v != nil {
		if s, ok := v.(string); ok {
			userID = s
		}
	}
	// Create child thread
	var thread *Thread
	var err error
	if userID != "" {
		thread, err = child.CreateThreadWithUser(ctx, userID, nil, metadata)
	} else {
		thread, err = child.CreateThread(ctx, nil, metadata)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to create child thread: %w", err)
	}
	// Add user message to child thread
	if userID != "" {
		_ = child.AddMessageWithUser(ctx, thread, *input, userID)
	} else {
		_ = child.AddMessage(ctx, thread, *input)
	}

	// Optional direct tool execution for deterministic child behavior in MAS tests
	if metadata != nil {
		if direct, ok := metadata["direct_tool"].(bool); ok && direct && len(child.Config.Tools) == 1 {
			tool := child.Config.Tools[0]
			args := map[string]interface{}{"text": input}
			res, err := tool.Function(ctx, args)
			if err != nil {
				return nil, fmt.Errorf("direct tool execution failed: %w", err)
			}
			// Convert result to string
			var result string
			switch v := res.(type) {
			case string:
				result = v
			case []byte:
				result = string(v)
			default:
				result = fmt.Sprintf("%v", v)
			}
			// Append assistant message to child thread for traceability
			assistant := Message{Role: "assistant", Content: result, CreatedAt: time.Now()}
			thread.Messages = append(thread.Messages, assistant)
			return map[string]interface{}{"agent": target, "thread_id": thread.ID, "result": result}, nil
		}
	}

	// Run the child agent via LLM
	run, err := child.CreateRun(ctx, thread)
	if err != nil {
		return nil, fmt.Errorf("failed to create child run: %w", err)
	}
	//repo := child.GetRepository()
	//if userID != "" && repo != nil {
	//	if err := child.ExecuteRunWithRepository(ctx, run, thread, userID, repo); err != nil {
	//		return nil, fmt.Errorf("child run failed: %w", err)
	//	}
	//} else {
	// 		if err := child.ExecuteRun(ctx, run, thread); err != nil {
	//			return nil, fmt.Errorf("child run failed: %w", err)
	//		}
	//}

	if err := child.ExecuteRun(ctx, run, thread); err != nil {
		return nil, fmt.Errorf("child run failed: %w", err)
	}

	// Extract last assistant message as the result
	var result string
	for i := len(thread.Messages) - 1; i >= 0; i-- {
		if thread.Messages[i].Role == "assistant" {
			result = fmt.Sprintf("%v", thread.Messages[i].Content)
			break
		}
	}
	return map[string]interface{}{
		"agent":     target,
		"thread_id": thread.ID,
		"result":    result,
	}, nil
}

// toMap safely converts any interface{} to map[string]interface{} if possible.
func toMap(v interface{}) map[string]interface{} {
	if v == nil {
		return nil
	}
	m, _ := v.(map[string]interface{})
	return m
}

// toMessage converts a generic interface{} (usually a map from tool JSON args) into a Message
// Supports either a pre-typed Message or a map[string]interface{} with fields:
// role (string), content (string), multiContent { type: "text"|"image_url", text: string, image_url: {url, detail} }
func toMessage(v interface{}) (Message, error) {
	if v == nil {
		return Message{}, errors.New("nil input")
	}
	// If already Message
	if msg, ok := v.(Message); ok {
		return msg, nil
	}
	m, ok := v.(map[string]interface{})
	if !ok {
		return Message{}, errors.New("input must be object")
	}
	var msg Message
	if role, _ := m["role"].(string); role != "" {
		msg.Role = role
	}
	if content, _ := m["content"].(string); content != "" {
		msg.Content = content
	}
	// Parse multiContent
	if mc, ok := m["multiContent"].(map[string]interface{}); ok {
		var part openai.ChatMessagePart
		if t, _ := mc["type"].(string); t != "" {
			part.Type = openai.ChatMessagePartType(t)
		}
		// Temporarily capture text, but if this part is an image_url we will move it to msg.Content
		var mcText string
		if txt, _ := mc["text"].(string); txt != "" {
			mcText = txt
			part.Text = txt
		}
		if iu, ok := mc["image_url"].(map[string]interface{}); ok {
			var img openai.ChatMessageImageURL
			if url, _ := iu["url"].(string); url != "" {
				img.URL = url
			}
			if det, _ := iu["detail"].(string); det != "" {
				// detail can be high/low/auto; store as-is
				img.Detail = openai.ImageURLDetail(det)
			}
			part.ImageURL = &img
		}
		// If this is an image_url part (or contains image_url), ensure type and move text to Content to avoid invalid image dict
		if part.ImageURL != nil && part.ImageURL.URL != "" {
			if part.Type == "" {
				part.Type = openai.ChatMessagePartTypeImageURL
			}
			if mcText != "" {
				if msg.Content == "" {
					msg.Content = mcText
				} else {
					msg.Content = msg.Content + "\n" + mcText
				}
				part.Text = "" // remove text from image part
			}
		}
		msg.MultiContent = part
	}
	// Defaults and normalization
	if msg.Role == "" {
		msg.Role = "user"
	}
	if msg.MultiContent.Type == "" && msg.MultiContent.ImageURL != nil && msg.MultiContent.ImageURL.URL != "" {
		msg.MultiContent.Type = "image_url"
	}
	if msg.MultiContent.ImageURL != nil && msg.MultiContent.ImageURL.URL != "" && msg.MultiContent.ImageURL.Detail == "" {
		msg.MultiContent.ImageURL.Detail = "low"
	}
	return msg, nil
}

// CreateThread creates a new conversation thread
func (a *Agent) CreateThread(ctx context.Context, messages []Message, metadata map[string]interface{}) (*Thread, error) {
	// Ensure all messages have a CreatedAt timestamp
	now := time.Now()
	system := Message{Role: "system", Content: a.systemPrompt}
	messages = append([]Message{system}, messages...)
	for i := range messages {
		if messages[i].CreatedAt.IsZero() {
			messages[i].CreatedAt = now
		}
	}

	thread := &Thread{
		ID:        generateID(),
		Messages:  messages,
		Metadata:  metadata,
		CreatedAt: time.Now(),
	}
	return thread, nil
}

// CreateThreadWithUser creates a new conversation thread with user identification
func (a *Agent) CreateThreadWithUser(ctx context.Context, userID string, messages []Message, metadata map[string]interface{}) (*Thread, error) {
	if userID == "" {
		return nil, errors.New("user ID cannot be empty")
	}

	// Create a new thread
	thread := &Thread{
		ID:        generateID(),
		Messages:  messages,
		Metadata:  metadata,
		CreatedAt: time.Now(),
	}

	// Add user ID to metadata
	if thread.Metadata == nil {
		thread.Metadata = make(map[string]interface{})
	}
	thread.Metadata["user_id"] = userID

	// Ensure all messages have a CreatedAt timestamp
	now := time.Now()
	for i := range messages {
		if messages[i].CreatedAt.IsZero() {
			messages[i].CreatedAt = now
		}
	}

	//// Save messages to repository if available
	//if a.repo != nil {
	//	repo, ok := a.repo.(interface {
	//		SaveMessage(ctx context.Context, userID string, role, content string, metadata map[string]interface{}, createdAt time.Time, threadID string) error
	//	})
	//	if ok {
	//		for _, msg := range messages {
	//			if err := repo.SaveMessage(ctx, userID, msg.Role, fmt.Sprintf("%v", msg.Content), msg.Metadata, msg.CreatedAt, thread.ID); err != nil {
	//				return nil, err
	//			}
	//		}
	//	}
	//}

	return thread, nil
}

// AddMessage adds a message to a thread
func (a *Agent) AddMessage(ctx context.Context, thread *Thread, message Message) error {
	if thread == nil {
		return errors.New("thread cannot be nil")
	}
	message.CreatedAt = time.Now()
	thread.Messages = append(thread.Messages, message)

	//// Save message to repository if available
	//if a.repo != nil && thread.Metadata != nil {
	//	userID, ok := thread.Metadata["user_id"].(string)
	//	if ok && userID != "" {
	//		repo, ok := a.repo.(interface {
	//			SaveMessage(ctx context.Context, userID string, role, content string, metadata map[string]interface{}, createdAt time.Time, threadID string) error
	//		})
	//		if ok {
	//			if err := repo.SaveMessage(ctx, userID, message.Role, fmt.Sprintf("%v", message.Content), message.Metadata, message.CreatedAt, thread.ID); err != nil {
	//				return err
	//			}
	//		}
	//	}
	//}

	return nil
}

// AddMessageWithUser adds a message to a thread with user identification
func (a *Agent) AddMessageWithUser(ctx context.Context, thread *Thread, message Message, userID string) error {
	if thread == nil {
		return errors.New("thread cannot be nil")
	}
	if userID == "" {
		return errors.New("user ID cannot be empty")
	}

	message.CreatedAt = time.Now()
	thread.Messages = append(thread.Messages, message)

	// Add user ID to metadata if not already present
	if thread.Metadata == nil {
		thread.Metadata = make(map[string]interface{})
	}
	thread.Metadata["user_id"] = userID

	// Save message to repository if available
	//if a.repo != nil {
	//	repo, ok := a.repo.(interface {
	//		SaveMessage(ctx context.Context, userID string, role, content string, metadata map[string]interface{}, createdAt time.Time, threadID string) error
	//	})
	//	if ok {
	//		if err := repo.SaveMessage(ctx, userID, message.Role, fmt.Sprintf("%v", message.Content), message.Metadata, message.CreatedAt, thread.ID); err != nil {
	//			return err
	//		}
	//	}
	//}

	return nil
}

// AddMessageWithoutSaving adds a message to a thread without saving to repository
func (a *Agent) AddMessageWithoutSaving(ctx context.Context, thread *Thread, message Message, userID string) error {
	if thread == nil {
		return errors.New("thread cannot be nil")
	}
	if userID == "" {
		return errors.New("user ID cannot be empty")
	}

	message.CreatedAt = time.Now()
	thread.Messages = append(thread.Messages, message)

	// Add user ID to metadata if not already present
	if thread.Metadata == nil {
		thread.Metadata = make(map[string]interface{})
	}
	thread.Metadata["user_id"] = userID

	// Do NOT save message to repository - this is the key difference

	return nil
}

// CreateRun creates a new run on a thread
func (a *Agent) CreateRun(ctx context.Context, thread *Thread) (*Run, error) {
	if thread == nil {
		return nil, errors.New("thread cannot be nil")
	}

	run := &Run{
		ID:        generateID(),
		ThreadID:  thread.ID,
		Status:    RunStatusQueued,
		Tools:     a.getToolsForRun(a.Config.Tools),
		CreatedAt: time.Now(),
	}
	return run, nil
}

// ExecuteRun executes a run synchronously
func (a *Agent) ExecuteRun(ctx context.Context, run *Run, thread *Thread) error {
	if run == nil {
		return errors.New("run cannot be nil")
	}
	if thread == nil {
		return errors.New("thread cannot be nil")
	}
	if a.client == nil {
		return errors.New("LLM client is not set")
	}

	now := time.Now()
	run.Status = RunStatusInProgress
	run.StartedAt = &now

	// Maximum number of steps to prevent infinite loops
	stepsRemaining := a.Config.MaxSteps

	for stepsRemaining > 0 {
		// Generate a response from the LLM
		assistantMessage, err := a.client.GenerateResponse(ctx, thread, run.Tools)
		if err != nil {
			run.Status = RunStatusFailed
			return err
		}

		// Add the assistant's message to the thread
		assistantMessage.CreatedAt = time.Now()
		thread.Messages = append(thread.Messages, *assistantMessage)

		// If there are no tool calls, we're done
		if len(assistantMessage.ToolCalls) == 0 {
			break
		}

		// Execute the tool calls - don't fail on tool errors, let LLM handle them
		ctxWithThread := context.WithValue(ctx, "current_thread", thread)
		err = a.client.ExecuteToolCalls(ctxWithThread, assistantMessage, run.Tools)
		if err != nil {
			// Log the error but don't fail the run - let LLM handle tool errors
			log.Printf("WARNING: Tool execution had errors, but continuing: %v", err)
		}

		// Create a new thread for the next iteration
		// This ensures that each tool message is preceded by an assistant message with tool_calls
		newThread := &Thread{
			ID:        thread.ID,
			Messages:  make([]Message, 0, len(thread.Messages)),
			Metadata:  thread.Metadata,
			CreatedAt: thread.CreatedAt,
		}

		// Copy all messages except the last assistant message
		for i := 0; i < len(thread.Messages)-1; i++ {
			newThread.Messages = append(newThread.Messages, thread.Messages[i])
		}

		// Add the last assistant message with tool calls
		newThread.Messages = append(newThread.Messages, *assistantMessage)

		// Add tool response messages to the thread
		for _, toolCall := range assistantMessage.ToolCalls {
			toolMessage := Message{
				Role:       "tool",
				Content:    toolCall.Output,
				ToolCallID: toolCall.ID,
				CreatedAt:  time.Now(),
				Metadata: map[string]interface{}{
					"tool_call_id": toolCall.ID,
				},
			}
			newThread.Messages = append(newThread.Messages, toolMessage)
		}

		// Replace the original thread with the new one
		*thread = *newThread

		stepsRemaining--

		// If we've reached the maximum number of steps, break
		if stepsRemaining == 0 {
			break
		}
	}

	endTime := time.Now()
	run.Status = RunStatusCompleted
	run.EndedAt = &endTime

	return nil
}

// ExecuteRunWithRepository executes a run synchronously and saves messages to repository
func (a *Agent) ExecuteRunWithRepository(ctx context.Context, run *Run, thread *Thread, userID string, repo interface{}) error {
	if run == nil {
		return errors.New("run cannot be nil")
	}
	if thread == nil {
		return errors.New("thread cannot be nil")
	}
	if a.client == nil {
		return errors.New("LLM client is not set")
	}
	if userID == "" {
		return errors.New("user ID cannot be empty")
	}

	now := time.Now()
	run.Status = RunStatusInProgress
	run.StartedAt = &now

	// Get the repository
	repository, ok := repo.(interface {
		SaveMessage(ctx context.Context, userID string, role, content string, metadata map[string]interface{}, createdAt time.Time, threadID string) error
	})
	if !ok {
		run.Status = RunStatusFailed
		log.Printf("Repository type assertion failed. Repository type: %T", repo)
		return errors.New("invalid repository")
	}

	// Maximum number of steps to prevent infinite loops
	stepsRemaining := a.Config.MaxSteps

	for stepsRemaining > 0 {
		// Generate a response from the LLM
		assistantMessage, err := a.client.GenerateResponse(ctx, thread, run.Tools)
		if err != nil {
			run.Status = RunStatusFailed
			return err
		}

		// Add the assistant's message to the thread
		assistantMessage.CreatedAt = time.Now()
		thread.Messages = append(thread.Messages, *assistantMessage)

		// Save assistant message to repository
		if err := repository.SaveMessage(ctx, userID, assistantMessage.Role, fmt.Sprintf("%v", assistantMessage.Content), assistantMessage.Metadata, assistantMessage.CreatedAt, thread.ID); err != nil {
			log.Printf("Error saving assistant message to repository: %v", err)
			// Continue execution even if saving fails
		}

		// If there are no tool calls, we're done
		if len(assistantMessage.ToolCalls) == 0 {
			break
		}

		// Execute the tool calls - don't fail on tool errors, let LLM handle them
		// Add userId to context for tool execution
		ctxWithUserID := context.WithValue(ctx, "user_id", userID)
		// Add current thread for tools to access parent messages
		ctxWithUserID = context.WithValue(ctxWithUserID, "current_thread", thread)
		err = a.client.ExecuteToolCalls(ctxWithUserID, assistantMessage, run.Tools)
		if err != nil {
			// Log the error but don't fail the run - let LLM handle tool errors
			log.Printf("WARNING: Tool execution had errors, but continuing: %v", err)
		}

		// Create a new thread for the next iteration
		// This ensures that each tool message is preceded by an assistant message with tool_calls
		newThread := &Thread{
			ID:        thread.ID,
			Messages:  make([]Message, 0, len(thread.Messages)),
			Metadata:  thread.Metadata,
			CreatedAt: thread.CreatedAt,
		}

		// Copy all messages except the last assistant message
		for i := 0; i < len(thread.Messages)-1; i++ {
			newThread.Messages = append(newThread.Messages, thread.Messages[i])
		}

		// Add the last assistant message with tool calls
		newThread.Messages = append(newThread.Messages, *assistantMessage)

		// Add tool response messages to the thread and save to repository
		for _, toolCall := range assistantMessage.ToolCalls {
			toolMessage := Message{
				Role:       "tool",
				Content:    toolCall.Output,
				ToolCallID: toolCall.ID,
				CreatedAt:  time.Now(),
				Metadata: map[string]interface{}{
					"tool_call_id": toolCall.ID,
				},
			}
			newThread.Messages = append(newThread.Messages, toolMessage)

			// Save tool message to repository
			if err := repository.SaveMessage(ctx, userID, toolMessage.Role, fmt.Sprintf("%v", toolMessage.Content), toolMessage.Metadata, toolMessage.CreatedAt, thread.ID); err != nil {
				log.Printf("Error saving tool message to repository: %v", err)
				// Continue execution even if saving fails
			}
		}

		// Replace the original thread with the new one
		*thread = *newThread

		stepsRemaining--

		// If we've reached the maximum number of steps, break
		if stepsRemaining == 0 {
			break
		}
	}

	endTime := time.Now()
	run.Status = RunStatusCompleted
	run.EndedAt = &endTime

	return nil
}

// StreamRun executes a run and streams the results
func (a *Agent) StreamRun(ctx context.Context, run *Run, thread *Thread, callback func(Message) error) error {
	if run == nil {
		return errors.New("run cannot be nil")
	}
	if thread == nil {
		return errors.New("thread cannot be nil")
	}
	if a.client == nil {
		return errors.New("LLM client is not set")
	}
	if callback == nil {
		return errors.New("callback function is required")
	}

	now := time.Now()
	run.Status = RunStatusInProgress
	run.StartedAt = &now

	// Maximum number of steps to prevent infinite loops
	stepsRemaining := a.Config.MaxSteps

	for stepsRemaining > 0 {
		// Generate a response from the LLM
		assistantMessage, err := a.client.GenerateResponse(ctx, thread, run.Tools)
		if err != nil {
			run.Status = RunStatusFailed
			return err
		}

		// Add the assistant's message to the thread
		assistantMessage.CreatedAt = time.Now()
		thread.Messages = append(thread.Messages, *assistantMessage)

		// Call the callback with the assistant's message
		if err := callback(*assistantMessage); err != nil {
			run.Status = RunStatusFailed
			return err
		}

		// If there are no tool calls, we're done
		if len(assistantMessage.ToolCalls) == 0 {
			break
		}

		// Execute the tool calls - don't fail on tool errors, let LLM handle them
		// Execute the tool calls - don't fail on tool errors, let LLM handle them
		// Attach current thread for tool validation (e.g., handoff parent tracking)
		ctxWithThread := context.WithValue(ctx, "current_thread", thread)
		err = a.client.ExecuteToolCalls(ctxWithThread, assistantMessage, run.Tools)
		if err != nil {
			// Log the error but don't fail the run - let LLM handle tool errors
			log.Printf("WARNING: Tool execution had errors, but continuing: %v", err)
		}

		// Create a new thread for the next iteration
		// This ensures that each tool message is preceded by an assistant message with tool_calls
		newThread := &Thread{
			ID:        thread.ID,
			Messages:  make([]Message, 0, len(thread.Messages)),
			Metadata:  thread.Metadata,
			CreatedAt: thread.CreatedAt,
		}

		// Copy all messages except the last assistant message
		for i := 0; i < len(thread.Messages)-1; i++ {
			newThread.Messages = append(newThread.Messages, thread.Messages[i])
		}

		// Add the last assistant message with tool calls
		newThread.Messages = append(newThread.Messages, *assistantMessage)

		// Add tool response messages to the thread and call the callback
		for _, toolCall := range assistantMessage.ToolCalls {
			toolMessage := Message{
				Role:       "tool",
				Content:    toolCall.Output,
				ToolCallID: toolCall.ID,
				CreatedAt:  time.Now(),
				Metadata: map[string]interface{}{
					"tool_call_id": toolCall.ID,
				},
			}
			newThread.Messages = append(newThread.Messages, toolMessage)

			// Call the callback with the tool message
			if err := callback(toolMessage); err != nil {
				run.Status = RunStatusFailed
				return err
			}
		}

		// Replace the original thread with the new one
		*thread = *newThread

		stepsRemaining--

		// If we've reached the maximum number of steps, break
		if stepsRemaining == 0 {
			break
		}
	}

	endTime := time.Now()
	run.Status = RunStatusCompleted
	run.EndedAt = &endTime

	return nil
}

// CancelRun cancels a running run
func (a *Agent) CancelRun(ctx context.Context, run *Run) error {
	if run == nil {
		return errors.New("run cannot be nil")
	}

	if run.Status != RunStatusInProgress && run.Status != RunStatusQueued {
		return errors.New("run is not in progress or queued")
	}

	run.Status = RunStatusCancelled
	endTime := time.Now()
	run.EndedAt = &endTime

	return nil
}

//// GetUserChatHistory retrieves the chat history for a user
//func (a *Agent) GetUserChatHistory(ctx context.Context, userID string, limit int) ([]Message, error) {
//	log.Printf("INFO: Agent.GetUserChatHistory called for user: %s, limit: %d", userID, limit)
//
//	if userID == "" {
//		log.Printf("ERROR: Agent.GetUserChatHistory - user ID cannot be empty")
//		return nil, errors.New("user ID cannot be empty")
//	}
//
//	//if a.repo == nil {
//	//	log.Printf("ERROR: Agent.GetUserChatHistory - repository is not set")
//	//	return nil, errors.New("repository is not set")
//	//}
//	//
//	//// Get the repository
//	//repo, ok := a.repo.(interface {
//	//	GetChatHistoryMap(ctx context.Context, userID string, limit int) ([]map[string]interface{}, error)
//	//})
//	//if !ok {
//	//	log.Printf("ERROR: Agent.GetUserChatHistory - repository does not implement GetChatHistoryMap method")
//	//	return nil, errors.New("repository does not implement GetChatHistoryMap method")
//	//}
//
//	//// Get chat history from repository
//	//chatMessages, err := repo.GetChatHistoryMap(ctx, userID, limit)
//	//if err != nil {
//	//	log.Printf("ERROR: Agent.GetUserChatHistory - failed to get chat history: %v", err)
//	//	return nil, err
//	//}
//
//	// Convert chat messages to agent messages
//	messages := make([]Message, 0, len(chatMessages))
//	for _, msg := range chatMessages {
//		role, _ := msg["role"].(string)
//		content := fmt.Sprintf("%v", msg["content"])
//		metadata, _ := msg["metadata"].(map[string]interface{})
//		createdAt, _ := msg["created_at"].(time.Time)
//
//		messages = append(messages, Message{
//			Role:      role,
//			Content:   content,
//			Metadata:  metadata,
//			CreatedAt: createdAt,
//		})
//	}
//
//	log.Printf("INFO: Agent.GetUserChatHistory successfully returned %d messages for user: %s", len(messages), userID)
//	return messages, nil
//}

//// GetThreadHistory retrieves the chat history for a specific thread
//func (a *Agent) GetThreadHistory(ctx context.Context, userID, threadID string) ([]Message, error) {
//	log.Printf("INFO: Agent.GetThreadHistory called for user: %s, thread: %s", userID, threadID)
//
//	if userID == "" {
//		log.Printf("ERROR: Agent.GetThreadHistory - user ID cannot be empty")
//		return nil, errors.New("user ID cannot be empty")
//	}
//	if threadID == "" {
//		log.Printf("ERROR: Agent.GetThreadHistory - thread ID cannot be empty")
//		return nil, errors.New("thread ID cannot be empty")
//	}
//
//	if a.repo == nil {
//		log.Printf("ERROR: Agent.GetThreadHistory - repository is not set")
//		return nil, errors.New("repository is not set")
//	}
//
//	// Get the repository
//	repo, ok := a.repo.(interface {
//		GetThreadHistoryMap(ctx context.Context, userID, threadID string) ([]map[string]interface{}, error)
//	})
//	if !ok {
//		log.Printf("ERROR: Agent.GetThreadHistory - repository does not implement GetThreadHistoryMap method")
//		return nil, errors.New("repository does not implement GetThreadHistoryMap method")
//	}
//
//	// Get thread history from repository
//	chatMessages, err := repo.GetThreadHistoryMap(ctx, userID, threadID)
//	if err != nil {
//		log.Printf("ERROR: Agent.GetThreadHistory - failed to get thread history: %v", err)
//		return nil, err
//	}
//
//	// Convert chat messages to agent messages
//	messages := make([]Message, 0, len(chatMessages))
//	for _, msg := range chatMessages {
//		role, _ := msg["role"].(string)
//		content := fmt.Sprintf("%v", msg["content"])
//		metadata, _ := msg["metadata"].(map[string]interface{})
//		createdAt, _ := msg["created_at"].(time.Time)
//
//		messages = append(messages, Message{
//			Role:      role,
//			Content:   content,
//			Metadata:  metadata,
//			CreatedAt: createdAt,
//		})
//	}
//
//	log.Printf("INFO: Agent.GetThreadHistory successfully returned %d messages for user: %s, thread: %s", len(messages), userID, threadID)
//	return messages, nil
//}
//
//// AnalyzeUserMessages analyzes previous user messages
//func (a *Agent) AnalyzeUserMessages(ctx context.Context, userID string, limit int) (string, error) {
//	log.Printf("INFO: Agent.AnalyzeUserMessages called for user: %s, limit: %d", userID, limit)
//
//	if userID == "" {
//		log.Printf("ERROR: Agent.AnalyzeUserMessages - user ID cannot be empty")
//		return "", errors.New("user ID cannot be empty")
//	}
//
//	// Get user chat history
//	messages, err := a.GetUserChatHistory(ctx, userID, limit)
//	if err != nil {
//		log.Printf("ERROR: Agent.AnalyzeUserMessages - failed to get user chat history: %v", err)
//		return "", err
//	}
//
//	if len(messages) == 0 {
//		log.Printf("INFO: Agent.AnalyzeUserMessages - no previous messages found for user: %s", userID)
//		return "No previous messages found for this user.", nil
//	}
//
//	// Filter user messages only
//	userMessages := make([]Message, 0)
//	for _, msg := range messages {
//		if msg.Role == "user" {
//			userMessages = append(userMessages, msg)
//		}
//	}
//
//	if len(userMessages) == 0 {
//		log.Printf("INFO: Agent.AnalyzeUserMessages - no previous user messages found for user: %s", userID)
//		return "No previous user messages found.", nil
//	}
//
//	// Create a summary of user messages
//	summary := "Previous user messages:\n"
//	for i, msg := range userMessages {
//		summary += fmt.Sprintf("%d. %s (at %s)\n", i+1, msg.Content, msg.CreatedAt.Format(time.RFC3339))
//	}
//
//	log.Printf("INFO: Agent.AnalyzeUserMessages successfully analyzed %d user messages for user: %s", len(userMessages), userID)
//	return summary, nil
//}
//
//// GetConversationContext automatically retrieves and formats conversation context for a user
//func (a *Agent) GetConversationContext(ctx context.Context, userID string, threadID string) (string, error) {
//	log.Printf("INFO: Agent.GetConversationContext called for user: %s, thread: %s", userID, threadID)
//
//	if userID == "" {
//		log.Printf("ERROR: Agent.GetConversationContext - user ID cannot be empty")
//		return "", errors.New("user ID cannot be empty")
//	}
//
//	var messages []Message
//	var err error
//
//	if threadID != "" {
//		// Get thread-specific history
//		log.Printf("INFO: Agent.GetConversationContext - getting thread-specific history")
//		messages, err = a.GetThreadHistory(ctx, userID, threadID)
//	} else {
//		// Get general user history (last 10 messages)
//		log.Printf("INFO: Agent.GetConversationContext - getting general user history")
//		messages, err = a.GetUserChatHistory(ctx, userID, 10)
//	}
//
//	if err != nil {
//		log.Printf("ERROR: Agent.GetConversationContext - failed to get conversation context: %v", err)
//		return "", fmt.Errorf("failed to get conversation context: %w", err)
//	}
//
//	if len(messages) == 0 {
//		log.Printf("INFO: Agent.GetConversationContext - no previous conversation context available for user: %s", userID)
//		return "No previous conversation context available.", nil
//	}
//
//	// Create a formatted context summary
//	context := "Conversation Context:\n"
//	context += "==================\n"
//
//	// Include last 5 messages for context
//	startIndex := 0
//	if len(messages) > 5 {
//		startIndex = len(messages) - 5
//	}
//
//	for i := startIndex; i < len(messages); i++ {
//		msg := messages[i]
//		context += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
//	}
//
//	context += "==================\n"
//	context += "Please use this context to provide relevant and contextual responses."
//
//	log.Printf("INFO: Agent.GetConversationContext successfully created context with %d messages for user: %s", len(messages[startIndex:]), userID)
//	return context, nil
//}

// Helper function to generate a unique ID
func generateID() string {
	// In a real implementation, this would generate a proper unique ID
	return "id_" + time.Now().Format("20060102150405")
}
