// src/utils/herbApi.js

const BASE_URL = 'https://steveo223-herb-ai-backend.hf.space';

/**
 * Fetches real-time structured plant metrics logged into the SQLite tables.
 * @returns {Promise<Array>} List of detected plant objects
 */
export async function fetchDetectedPlants() {
  try {
    const response = await fetch(`${BASE_URL}/api/telemetry`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const result = await response.json();
    return result.data || []; // Returns the clean telemetry array
  } catch (error) {
    console.error("Failed to connect to FastAPI telemetry route:", error);
    return [];
  }
}

/**
 * Kicks off the asynchronous computer vision frame-parsing background pipeline.
 * @returns {Promise<boolean>} Success status
 */
export async function triggerVisionScan() {
  try {
    const response = await fetch(`${BASE_URL}/api/scan`, { method: 'POST' });
    return response.ok;
  } catch (error) {
    console.error("Failed to trigger backend computer vision scan:", error);
    return false;
  }
}

/**
 * @param {string} userQuestion - The question to ask the agent
 * @returns {Promise<string>} Grounded medical text response
 */
export async function askBotanicalQuestion(userQuestion) {
  try {
    const response = await fetch(`${BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: userQuestion }),
    });
    
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const result = await response.json();
    return result.answer; 
  } catch (error) {
    console.error("Chat routing query engine mapping failed:", error);
    return "Error generating response from the RAG query server.";
  }
}