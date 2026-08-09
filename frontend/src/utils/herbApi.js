// frontend/src/utils/herbApi.js

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "https://steveo223-herb-ai-backend.hf.space";

/**
 * Fetches real-time telemetry from SQLite database.
 */
export async function fetchDetectedPlants() {
  try {
    const response = await fetch(`${BASE_URL}/api/telemetry`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const result = await response.json();
    return result.data || [];
  } catch (error) {
    console.error("Telemetry fetch error:", error);
    return [];
  }
}

/**
 * Triggers video file tracking scan.
 * @param {File} videoFile - The uploaded video file object.
 */
export async function triggerVisionScan(videoFile) {
  try {
    const formData = new FormData();
    if (videoFile) {
      formData.append("file", videoFile);
    }

    const response = await fetch(`${BASE_URL}/api/scan`, {
      method: "POST",
      body: formData,
    });

    // Capture the exact FastAPI 422 error reason
    if (!response.ok) {
      const errorDetails = await response.json();
      console.error("🚨 FASTAPI 422 ERROR DETAILS:", JSON.stringify(errorDetails, null, 2));
      return false;
    }
    
    return true;
  } catch (error) {
    console.error("Vision scan trigger failed:", error);
    return false;
  }
}
/**
 * Uploads a static image for botanical identification.
 * @param {File} imageFile - The uploaded image file object.
 */
export async function uploadImage(imageFile) {
  try {
    const formData = new FormData();
    formData.append("file", imageFile);

    const response = await fetch(`${BASE_URL}/api/upload-image`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("Image upload failed:", error);
    return null;
  }
}

/**
 * Sends a clinical inquiry to the RAG LLM query engine.
 * @param {string} userQuestion - The question text.
 */
export async function askBotanicalQuestion(userQuestion) {
  try {
    const response = await fetch(`${BASE_URL}/api/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query_text: userQuestion }),
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const result = await response.json();
    return result.response || result.answer;
  } catch (error) {
    console.error("Query engine failed:", error);
    return "Error generating response from the RAG query server.";
  }
}