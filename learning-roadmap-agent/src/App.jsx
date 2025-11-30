import React, { useState, useEffect } from "react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  BookOpen,
  MessageSquare,
  Brain,
  Map,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Loader2,
  AlertCircle,
  TrendingUp,
  Sparkles,
  Network,
} from "lucide-react";

const LearningRoadmapAgent = () => {
  // Main state
  const [topic, setTopic] = useState("");
  const [roadmap, setRoadmap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentView, setCurrentView] = useState("input");
  const [currentNode, setCurrentNode] = useState(null);

  // Visual roadmap state
  const [visualRoadmap, setVisualRoadmap] = useState(null);
  const [showVisualRoadmap, setShowVisualRoadmap] = useState(false);
  const [generatingVisual, setGeneratingVisual] = useState(false);

  // Chat state
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatSessionId, setChatSessionId] = useState(null);

  // Quiz state
  const [currentQuiz, setCurrentQuiz] = useState(null);
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizResults, setQuizResults] = useState(null);
  const [quizPrediction, setQuizPrediction] = useState(null);
  const [showPrediction, setShowPrediction] = useState(false);

  // UI state
  const [expandedNodes, setExpandedNodes] = useState({});

  // ML/User data tracking
  const [userData, setUserData] = useState({
    resourcesViewed: [],
    chatMessages: [],
    quizResults: [],
    timeTracking: {},
  });
  const [userLearningStyle, setUserLearningStyle] = useState(null);
  const [rankedResources, setRankedResources] = useState({});

  const ML_API_URL = "http://localhost:5000";

  // Render visual roadmap when available
  useEffect(() => {
    if (visualRoadmap && showVisualRoadmap) {
      const renderMermaid = async () => {
        try {
          // Dynamically load Mermaid from CDN
          if (!window.mermaid) {
            const script = document.createElement("script");
            script.src =
              "https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.6.1/mermaid.min.js";
            script.async = true;
            script.onload = () => {
              window.mermaid.initialize({
                startOnLoad: false,
                theme: "default",
                securityLevel: "loose",
              });
              renderDiagram();
            };
            document.head.appendChild(script);
          } else {
            renderDiagram();
          }

          function renderDiagram() {
            const element = document.getElementById("mermaid-diagram");
            if (element && window.mermaid) {
              element.innerHTML = visualRoadmap;
              window.mermaid.run({ nodes: [element] });
            }
          }
        } catch (error) {
          console.error("Mermaid rendering error:", error);
        }
      };
      renderMermaid();
    }
  }, [visualRoadmap, showVisualRoadmap]);

  const roadmapSchema = {
    type: "object",
    properties: {
      title: { type: "string" },
      description: { type: "string" },
      estimatedTime: { type: "string" },
      nodes: {
        type: "array",
        items: {
          type: "object",
          properties: {
            id: { type: "string" },
            title: { type: "string" },
            description: { type: "string" },
            prerequisites: { type: "array", items: { type: "string" } },
            resources: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  title: { type: "string" },
                  url: { type: "string" },
                  type: { type: "string" },
                  difficulty: { type: "string" },
                  tags: { type: "array", items: { type: "string" } },
                },
              },
            },
            keyPoints: { type: "array", items: { type: "string" } },
          },
          required: ["id", "title", "description", "resources", "keyPoints"],
        },
      },
    },
    required: ["title", "description", "estimatedTime", "nodes"],
  };

  const quizSchema = {
    type: "object",
    properties: {
      nodeId: { type: "string" },
      nodeTitle: { type: "string" },
      questions: {
        type: "array",
        items: {
          type: "object",
          properties: {
            id: { type: "string" },
            question: { type: "string" },
            options: {
              type: "array",
              items: { type: "string" },
              minItems: 4,
              maxItems: 4,
            },
            correct: { type: "integer", minimum: 0, maximum: 3 },
            explanation: { type: "string" },
          },
          required: ["id", "question", "options", "correct", "explanation"],
        },
      },
    },
    required: ["nodeId", "nodeTitle", "questions"],
  };

  const generateRoadmap = async () => {
    if (!topic.trim()) return;

    setLoading(true);
    setRoadmap(null);
    setCurrentView("input");
    setCurrentQuiz(null);
    setVisualRoadmap(null);
    setShowVisualRoadmap(false);
    setRankedResources({}); // Reset ranked resources

    try {
      const response = await fetch("http://localhost:5000/generate-roadmap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          prompt: `Create a comprehensive learning roadmap for: ${topic}. 

Research and reference roadmap.sh learning paths for structural inspiration. Study how they organize topics progressively.

Create 6-10 nodes in logical learning order. For each resource, include:
- Real, working URLs from reputable sources (official docs, MDN, freeCodeCamp, Udemy, YouTube channels, etc.)
- Resource type (video, article, documentation, interactive, tutorial, course)
- Difficulty level (beginner, intermediate, advanced)
- Relevant tags for categorization (e.g., ["hands-on", "official", "free", "comprehensive"])

Make it practical, actionable, and similar in structure to roadmap.sh paths.`,
          schema: roadmapSchema,
        }),
      });

      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      setRoadmap(data);
      setCurrentView("roadmap");

      // Rank resources immediately (works even without learning style now)
      console.log("🔄 Starting resource ranking...");
      await rankAllResources(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to generate roadmap. Make sure Flask backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const rankAllResources = async (roadmapData) => {
    // FIXED: Now ranks resources even without learning style
    try {
      const ranked = {};
      const learningStyle = userLearningStyle || "Visual"; // Default to Visual

      for (const node of roadmapData.nodes) {
        const response = await fetch(`${ML_API_URL}/api/rank-resources`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            resources: node.resources,
            topic: node.title + " " + node.description,
            learning_style: learningStyle,
            difficulty_level: 0.5,
          }),
        });

        if (response.ok) {
          const data = await response.json();
          if (data.ranked) {
            ranked[node.id] = data.recommendations;
            console.log(`✅ Ranked resources for ${node.title}`);
          }
        }
      }

      if (Object.keys(ranked).length > 0) {
        setRankedResources(ranked);
        console.log(
          `✅ Successfully ranked resources for ${
            Object.keys(ranked).length
          } nodes`
        );
      }
    } catch (error) {
      console.log("Resource ranking not available:", error);
    }
  };

  const generateVisualRoadmap = async () => {
    if (!roadmap) return;

    setGeneratingVisual(true);

    try {
      const response = await fetch(
        "http://localhost:5000/generate-visual-roadmap",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            roadmap: roadmap,
          }),
        }
      );

      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      setVisualRoadmap(data.mermaidCode);
      setShowVisualRoadmap(true);
    } catch (error) {
      console.error("Error generating visual roadmap:", error);
      alert("Failed to generate visual roadmap.");
    } finally {
      setGeneratingVisual(false);
    }
  };

  const requestQuizPrediction = async () => {
    if (!currentNode) return;

    try {
      const response = await fetch(`${ML_API_URL}/api/predict-quiz-score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          previous_scores: userData.quizResults.map((q) => q.score),
          topic_difficulty: 0.7,
          days_since_last_quiz: calculateDaysSinceLastQuiz(),
          total_study_time: 60,
          chat_messages_sent: userData.chatMessages.filter(
            (m) => m.nodeId === currentNode.id
          ).length,
          resources_viewed: userData.resourcesViewed.filter(
            (r) => r.nodeId === currentNode.id
          ).length,
          learning_style: userLearningStyle || "Visual",
        }),
      });

      if (response.ok) {
        const predictionData = await response.json();
        setQuizPrediction(predictionData);
        setShowPrediction(true);
      }
    } catch (error) {
      console.log("ML prediction not available:", error);
      alert(
        "Prediction service unavailable. Make sure Flask backend is running."
      );
    }
  };

  const startChat = async (node) => {
    setCurrentNode(node);

    const systemInstruction = `You are an expert tutor helping someone learn about "${
      node.title
    }" as part of their ${roadmap.title} journey. 

Context: ${node.description}
Key points they should understand: ${node.keyPoints.join(", ")}

Provide clear, educational responses. Use examples when helpful. Be encouraging and supportive.`;

    try {
      const response = await fetch("http://localhost:5000/start-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          systemInstruction: systemInstruction,
        }),
      });

      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      setChatSessionId(data.sessionId);
      setChatHistory([
        {
          role: "assistant",
          content: `Let's dive into "${node.title}"! I can help explain concepts, answer questions, provide examples, or discuss best practices. What would you like to know?`,
        },
      ]);
      setCurrentView("chat");
      setCurrentQuiz(null);
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to start chat.");
    }
  };

  const sendMessage = async () => {
    if (!chatInput.trim() || chatLoading || !chatSessionId) return;

    const userMsg = chatInput;
    setChatInput("");
    setChatHistory((prev) => [...prev, { role: "user", content: userMsg }]);
    setChatLoading(true);

    setUserData((prev) => ({
      ...prev,
      chatMessages: [
        ...prev.chatMessages,
        { nodeId: currentNode.id, message: userMsg, timestamp: Date.now() },
      ],
    }));

    try {
      const response = await fetch("http://localhost:5000/send-message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: chatSessionId, message: userMsg }),
      });

      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      setChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: data.text },
      ]);
    } catch (error) {
      console.error("Error:", error);
      setChatHistory((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, there was an error. Please try again.",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleStartQuiz = async (node) => {
    setCurrentNode(node);
    setShowPrediction(false);
    setQuizPrediction(null);
    generateQuiz(node);
  };

  const generateQuiz = async (node) => {
    setLoading(true);
    setCurrentQuiz(null);
    setQuizAnswers({});
    setQuizResults(null);

    try {
      const response = await fetch("http://localhost:5000/generate-quiz", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          prompt: `Create exactly 5 practical multiple choice questions for:
Title: ${node.title}
Description: ${node.description}
Key Points: ${node.keyPoints.join(", ")}

Make questions test real understanding.`,
          schema: {
            ...quizSchema,
            properties: {
              ...quizSchema.properties,
              nodeId: { type: "string", default: node.id },
              nodeTitle: { type: "string", default: node.title },
            },
          },
        }),
      });

      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      setCurrentQuiz(data);
      setCurrentView("quiz");
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to generate quiz.");
    } finally {
      setLoading(false);
    }
  };

  const submitQuiz = () => {
    const results = currentQuiz.questions.map((q) => ({
      question: q.question,
      selectedAnswer: quizAnswers[q.id],
      correctAnswer: q.correct,
      isCorrect: quizAnswers[q.id] === q.correct,
      explanation: q.explanation,
    }));

    const score = results.filter((r) => r.isCorrect).length / results.length;
    setQuizResults({ results, score, total: currentQuiz.questions.length });

    setUserData((prev) => ({
      ...prev,
      quizResults: [
        ...prev.quizResults,
        { nodeId: currentNode.id, score: score, timestamp: Date.now() },
      ],
    }));
  };

  const toggleNode = (nodeId) => {
    setExpandedNodes((prev) => ({ ...prev, [nodeId]: !prev[nodeId] }));
  };

  const trackResourceView = (resource, nodeId) => {
    setUserData((prev) => ({
      ...prev,
      resourcesViewed: [
        ...prev.resourcesViewed,
        {
          resourceId: resource.title,
          nodeId: nodeId,
          type: resource.type,
          timestamp: Date.now(),
        },
      ],
    }));
  };

  const calculateDaysSinceLastQuiz = () => {
    if (userData.quizResults.length === 0) return 1;
    const lastQuiz = userData.quizResults[userData.quizResults.length - 1];
    const daysSince = Math.floor(
      (Date.now() - lastQuiz.timestamp) / (1000 * 60 * 60 * 24)
    );
    return Math.max(daysSince, 1);
  };

  useEffect(() => {
    const analyzeUser = async () => {
      if (
        userData.resourcesViewed.length >= 3 ||
        userData.quizResults.length >= 1
      ) {
        try {
          const response = await fetch(
            `${ML_API_URL}/api/analyze-learning-style`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(userData),
            }
          );

          if (response.ok) {
            const analysis = await response.json();
            setUserLearningStyle(analysis.learning_style);

            // Re-rank resources when learning style is detected
            if (roadmap) {
              await rankAllResources(roadmap);
            }
          }
        } catch (error) {
          console.log("ML analysis not available:", error);
        }
      }
    };

    analyzeUser();
  }, [userData.resourcesViewed.length, userData.quizResults.length]);

  const getResourcesToDisplay = (node) => {
    // Use ranked resources if available, otherwise use original
    return rankedResources[node.id] || node.resources;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Brain className="w-10 h-10 text-indigo-600" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                AI Learning Roadmap Agent
              </h1>
            </div>
            {userLearningStyle && (
              <div className="px-4 py-2 bg-purple-100 text-purple-800 rounded-full font-medium flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                {userLearningStyle} Learner
              </div>
            )}
          </div>
          <p className="text-gray-600 mb-6">
            Enter any topic, get a personalized roadmap with resources,
            interactive discussions, quizzes, and visual diagrams!
          </p>

          {currentView === "input" && (
            <div className="flex gap-3">
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && generateRoadmap()}
                placeholder="e.g., React.js, Machine Learning, Python, Web Development..."
                className="flex-1 px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-indigo-500 transition"
              />
              <button
                onClick={generateRoadmap}
                disabled={loading || !topic.trim()}
                className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Map className="w-5 h-5" />
                    Generate Roadmap
                  </>
                )}
              </button>
            </div>
          )}

          {currentView !== "input" && (
            <div className="flex gap-3">
              <button
                onClick={() => setCurrentView("input")}
                className="px-4 py-2 text-indigo-600 hover:bg-indigo-50 rounded-lg transition"
              >
                ← New Topic
              </button>

              {roadmap && !showVisualRoadmap && (
                <button
                  onClick={generateVisualRoadmap}
                  disabled={generatingVisual}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 transition flex items-center gap-2"
                >
                  {generatingVisual ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Generating Diagram...
                    </>
                  ) : (
                    <>
                      <Network className="w-4 h-4" />
                      View Visual Roadmap
                    </>
                  )}
                </button>
              )}

              {showVisualRoadmap && (
                <button
                  onClick={() => setShowVisualRoadmap(false)}
                  className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
                >
                  Hide Diagram
                </button>
              )}
            </div>
          )}
        </div>

        {/* Visual Roadmap Display */}
        {showVisualRoadmap && visualRoadmap && (
          <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Network className="w-6 h-6 text-green-600" />
              Visual Learning Path
            </h3>
            <div className="bg-gray-50 rounded-xl p-6 overflow-x-auto">
              <div id="mermaid-diagram" className="flex justify-center"></div>
            </div>
          </div>
        )}

        {/* Roadmap Display */}
        {currentView === "roadmap" && roadmap && (
          <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
            <h2 className="text-3xl font-bold text-gray-800 mb-2">
              {roadmap.title}
            </h2>
            <p className="text-gray-600 mb-2">{roadmap.description}</p>
            <p className="text-sm text-indigo-600 font-medium mb-6">
              Estimated Time: {roadmap.estimatedTime}
            </p>

            <div className="space-y-4">
              {roadmap.nodes.map((node, idx) => (
                <div
                  key={node.id}
                  className="border-2 border-gray-200 rounded-xl overflow-hidden hover:border-indigo-300 transition"
                >
                  <div
                    className="bg-gradient-to-r from-indigo-50 to-purple-50 p-4 cursor-pointer"
                    onClick={() => toggleNode(node.id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center font-bold">
                            {idx + 1}
                          </span>
                          <h3 className="text-xl font-bold text-gray-800">
                            {node.title}
                          </h3>
                        </div>
                        <p className="text-gray-600 ml-11">
                          {node.description}
                        </p>
                      </div>
                      {expandedNodes[node.id] ? (
                        <ChevronUp className="w-6 h-6 text-gray-400" />
                      ) : (
                        <ChevronDown className="w-6 h-6 text-gray-400" />
                      )}
                    </div>
                  </div>

                  {expandedNodes[node.id] && (
                    <div className="p-4 bg-white">
                      {node.prerequisites && node.prerequisites.length > 0 && (
                        <div className="mb-4">
                          <p className="text-sm font-medium text-gray-700 mb-1">
                            Prerequisites:
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {node.prerequisites.map((pre, i) => (
                              <span
                                key={i}
                                className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs rounded"
                              >
                                {pre}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="mb-4">
                        <p className="text-sm font-medium text-gray-700 mb-2">
                          Key Learning Points:
                        </p>
                        <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                          {node.keyPoints.map((point, i) => (
                            <li key={i}>{point}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="mb-4">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-sm font-medium text-gray-700">
                            Resources
                          </p>
                          {rankedResources[node.id] && (
                            <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full font-medium flex items-center gap-1">
                              <Sparkles className="w-3 h-3" />
                              AI Ranked for You
                            </span>
                          )}
                        </div>
                        <div className="space-y-2">
                          {getResourcesToDisplay(node).map((res, i) => (
                            <a
                              key={i}
                              href={res.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={() => trackResourceView(res, node.id)}
                              className="flex items-center gap-2 p-2 bg-gray-50 rounded hover:bg-gray-100 transition"
                            >
                              <BookOpen className="w-4 h-4 text-indigo-600" />
                              <span className="flex-1 text-sm text-gray-700">
                                {res.title}
                              </span>
                              <span className="text-xs text-gray-500 px-2 py-1 bg-white rounded">
                                {res.type}
                              </span>
                              {res.relevance_score && (
                                <span className="text-xs text-purple-600 font-medium">
                                  {Math.round(res.relevance_score * 100)}% match
                                </span>
                              )}
                              <ExternalLink className="w-4 h-4 text-gray-400" />
                            </a>
                          ))}
                        </div>
                      </div>

                      <div className="flex gap-3">
                        <button
                          onClick={() => startChat(node)}
                          className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition flex items-center justify-center gap-2"
                        >
                          <MessageSquare className="w-4 h-4" />
                          Discuss This Topic
                        </button>
                        <button
                          onClick={() => handleStartQuiz(node)}
                          className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition flex items-center justify-center gap-2"
                        >
                          <Brain className="w-4 h-4" />
                          Take Quiz
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Chat Mode */}
        {currentView === "chat" && currentNode && (
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-2xl font-bold text-gray-800">
                  {currentNode.title}
                </h3>
                <p className="text-sm text-gray-600">Interactive Discussion</p>
              </div>
              <button
                onClick={() => setCurrentView("roadmap")}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition"
              >
                Back to Roadmap
              </button>
            </div>

            <div className="bg-gray-50 rounded-xl p-4 mb-4 h-96 overflow-y-auto space-y-4">
              {chatHistory.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-lg ${
                      msg.role === "user"
                        ? "bg-indigo-600 text-white"
                        : "bg-white text-gray-800 border border-gray-200"
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  </div>
                </div>
              ))}
              {chatLoading && (
                <div className="flex justify-start">
                  <div className="bg-white text-gray-800 border border-gray-200 p-3 rounded-lg flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Thinking...
                  </div>
                </div>
              )}
            </div>

            <div className="flex gap-3">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Ask a question or discuss the topic..."
                className="flex-1 px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-indigo-500 transition"
              />
              <button
                onClick={sendMessage}
                disabled={chatLoading || !chatInput.trim()}
                className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-300 transition"
              >
                Send
              </button>
            </div>
          </div>
        )}

        {/* Quiz Mode */}
        {currentView === "quiz" && (
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-2xl font-bold text-gray-800">
                  {currentQuiz
                    ? `${currentQuiz.nodeTitle} - Quiz`
                    : "Loading Quiz..."}
                </h3>
                <p className="text-sm text-gray-600">Test your knowledge!</p>
              </div>
              <button
                onClick={() => setCurrentView("roadmap")}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition"
              >
                Back to Roadmap
              </button>
            </div>
            {/* Prediction Button */}
            {currentNode && !quizResults && !showPrediction && (
              <button
                onClick={requestQuizPrediction}
                className="w-full px-4 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg font-medium hover:from-purple-700 hover:to-indigo-700 transition flex items-center justify-center gap-2 mb-4"
              >
                <TrendingUp className="w-5 h-5" />
                Predict My Quiz Score
              </button>
            )}

            {quizPrediction && !quizResults && (
              <div className="bg-indigo-50 rounded-xl p-4 mb-6 border-2 border-indigo-200">
                <div className="flex items-center gap-3 mb-2">
                  <AlertCircle className="w-6 h-6 text-indigo-600" />
                  <h4 className="font-bold text-indigo-900">
                    AI Performance Prediction
                  </h4>
                </div>
                <p className="text-indigo-800 mb-2">
                  Based on your learning behavior, we predict you'll score
                  around{" "}
                  <span className="font-bold text-2xl">
                    {Math.round(quizPrediction.predicted_score * 100)}%
                  </span>
                </p>
                <p className="text-sm text-indigo-700 italic">
                  {quizPrediction.advice}
                </p>
              </div>
            )}

            {!currentQuiz && loading && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
              </div>
            )}

            {currentQuiz && !quizResults && (
              <>
                <div className="space-y-6 mb-6">
                  {currentQuiz.questions.map((q, idx) => (
                    <div
                      key={q.id}
                      className="border-2 border-gray-200 rounded-xl p-4"
                    >
                      <div className="font-medium text-gray-800 mb-3">
                        {idx + 1}.{" "}
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {q.question}
                        </ReactMarkdown>
                      </div>

                      <div className="space-y-2">
                        {q.options.map((opt, optIdx) => (
                          <label
                            key={optIdx}
                            className="flex items-center gap-3 p-3 border-2 border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50 transition"
                          >
                            <input
                              type="radio"
                              name={q.id}
                              checked={quizAnswers[q.id] === optIdx}
                              onChange={() =>
                                setQuizAnswers({
                                  ...quizAnswers,
                                  [q.id]: optIdx,
                                })
                              }
                              className="w-4 h-4"
                            />
                            <div className="text-gray-700">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {opt}
                              </ReactMarkdown>
                            </div>
                          </label>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  onClick={submitQuiz}
                  disabled={
                    Object.keys(quizAnswers).length !==
                    currentQuiz.questions.length
                  }
                  className="w-full px-6 py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
                >
                  Submit Quiz
                </button>
              </>
            )}

            {quizResults && (
              <div>
                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 mb-6">
                  <h4 className="text-2xl font-bold text-gray-800 mb-2">
                    Your Score: {quizResults.score} / {quizResults.total}
                  </h4>
                  <p className="text-gray-600">
                    {quizResults.score === quizResults.total
                      ? "🎉 Perfect! You have mastered this topic!"
                      : quizResults.score >= quizResults.total * 0.7
                      ? "👍 Good job! Review the explanations below to strengthen your understanding."
                      : "📚 Keep learning! Review the material and try again."}
                  </p>
                </div>

                <div className="space-y-4 mb-6">
                  {quizResults.results.map((result, idx) => (
                    <div
                      key={idx}
                      className={`border-2 rounded-xl p-4 ${
                        result.isCorrect
                          ? "border-green-300 bg-green-50"
                          : "border-red-300 bg-red-50"
                      }`}
                    >
                      <p className="font-medium text-gray-800 mb-2">
                        {result.question}
                      </p>
                      <p className="text-sm mb-1">
                        <span className="font-medium">Your answer: </span>
                        {
                          currentQuiz.questions[idx].options[
                            result.selectedAnswer
                          ]
                        }
                      </p>
                      {!result.isCorrect && (
                        <p className="text-sm mb-1">
                          <span className="font-medium">Correct answer: </span>
                          {
                            currentQuiz.questions[idx].options[
                              result.correctAnswer
                            ]
                          }
                        </p>
                      )}
                      <div className="text-sm text-gray-700 mt-2 italic prose">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {result.explanation}
                        </ReactMarkdown>
                      </div>
                    </div>
                  ))}
                </div>

                <button
                  onClick={() => {
                    setCurrentView("roadmap");
                    setQuizResults(null);
                    setQuizAnswers({});
                    setQuizPrediction(null);
                  }}
                  className="w-full px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition"
                >
                  Back to Roadmap
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default LearningRoadmapAgent;
