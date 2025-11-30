// services/claudeAPI.js - EXISTING (KEEP THIS)

export const generateRoadmap = async (topic) => {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4000,
      messages: [
        {
          role: "user",
          content: `Create learning roadmap for: ${topic}`,
        },
      ],
    }),
  });

  const data = await response.json();
  const text = data.content.find((c) => c.type === "text")?.text || "";
  const cleaned = text.replace(/```json\n?|```\n?/g, "").trim();
  return JSON.parse(cleaned);
};

export const sendChatMessage = async (messages, nodeContext) => {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 2000,
      messages: messages,
      system: `You are tutoring: ${nodeContext.title}`,
    }),
  });

  const data = await response.json();
  return data.content.find((c) => c.type === "text")?.text || "";
};

export const generateQuiz = async (node) => {
  // Same as before - uses Claude API
  // ...
};
