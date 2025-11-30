// services/mlAPI.js - NEW (ADD THIS)

const ML_API_URL = "http://localhost:5000";

export const analyzeUserBehavior = async (userData) => {
  const response = await fetch(`${ML_API_URL}/api/analyze-learning-style`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(userData),
  });

  return response.json();
  // Returns: { learningStyle: 'Visual', confidence: {...} }
};

export const predictQuizScore = async (quizData) => {
  const response = await fetch(`${ML_API_URL}/api/predict-quiz-score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(quizData),
  });

  return response.json();
  // Returns: { predicted_score: 0.75, advice: "..." }
};

export const rankResources = async (rankingData) => {
  const response = await fetch(`${ML_API_URL}/api/rank-resources`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rankingData),
  });

  return response.json();
  // Returns: { recommendations: [...] }
};

export const getStudyAdvice = async (userData) => {
  const response = await fetch(`${ML_API_URL}/api/get-study-advice`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(userData),
  });

  return response.json();
};
