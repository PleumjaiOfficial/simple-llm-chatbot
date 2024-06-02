{
  lc: 1,
  type: "constructor",
  id: ["langchain", "schema", "runnable", "RunnableSequence"],
  kwargs: {
    first: {
      lc: 1,
      type: "constructor",
      id: ["langchain", "prompts", "prompt", "PromptTemplate"],
      kwargs: {
        input_variables: ["productDesc"],
        template_format: "f-string",
        template:
          "Generate a promotional tweet for a product, from this product description: {productDesc}",
        partial_variables: undefined,
      },
    },
    last: {
      lc: 1,
      type: "constructor",
      id: ["langchain", "chat_models", "openai", "ChatOpenAI"],
      kwargs: {
        callbacks: undefined,
        openai_api_key: { lc: 1, type: "secret", id: ["OPENAI_API_KEY"] },
        verbose: undefined,
      },
    },
  },
};