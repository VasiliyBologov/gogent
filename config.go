package gogent

type Config struct {
	OpenAI OpenAIConfig
}

type OpenAIConfig struct {
	APIKey          string
	VisionModelName string
	ThinkingModel   string
}
