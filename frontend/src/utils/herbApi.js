// frontend/src/utils/herbApi.js

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://steveo223-herb-ai-backend.hf.space";

const getSessionId = () => {
  let sessionId = sessionStorage.getItem("herb_session_id");
  if (!sessionId) {
    if (window.crypto && window.crypto.randomUUID) {
      sessionId = window.crypto.randomUUID();
    } else {
      sessionId = "session_" + Math.random().toString(36).substring(2, 15);
    }
    sessionStorage.setItem("herb_session_id", sessionId);
  }
  return sessionId;
};
/**
 * Fetches real-time telemetry from SQLite database.
 */
export async function fetchDetectedPlants() {
  try {
    const response = await fetch(
      `${BASE_URL}/api/telemetry?session_id=${getSessionId()}`,
    );
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const result = await response.json();
    return result.data || [];
  } catch (error) {
    console.error("Telemetry fetch error:", error);
    return [];
  }
}

/**
 * Checks the true execution state of the backend pipeline.
 */
export async function checkScanStatus() {
  try {
    const response = await fetch(`${BASE_URL}/api/scan-status?session_id=${getSessionId()}`);
    if (!response.ok) return { is_scanning: false };
    return await response.json();
  } catch (error) {
    console.error("Failed to check scan status:", error);
    return { is_scanning: false };
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

    formData.append("session_id", getSessionId());

    const response = await fetch(`${BASE_URL}/api/scan`, {
      method: "POST",
      body: formData,
    });

    // Capture the exact FastAPI 422 error reason
    if (!response.ok) {
      const errorDetails = await response.json();
      console.error(
        "🚨 FASTAPI 422 ERROR DETAILS:",
        JSON.stringify(errorDetails, null, 2),
      );
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
    formData.append("session_id", getSessionId());

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
 * Sends a static image to the explicit predict endpoint for botanical identification.
 * @param {File} imageFile - The uploaded image file object.
 */
export async function predictPlantImage(imageFile) {
  try {
    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("session_id", getSessionId());

    const response = await fetch(`${BASE_URL}/api/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Image prediction failed:", error);
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

/**
 * Streams the RAG LLM query response. Includes auto-resume fallback logic for
 * when mobile browsers suspend network connections during backgrounding.
 * @param {string} userQuestion - The question text.
 * @param {function} onChunk - Callback function handling chunks and replacements.
 */
export async function streamBotanicalQuestion(userQuestion, onChunk) {
  try {
    const response = await fetch(`${BASE_URL}/api/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query_text: userQuestion }),
    });

    if (!response.ok) throw new Error("Stream connection failed");

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let done = false;

    while (!done) {
      const { value, done: readerDone } = await reader.read();
      done = readerDone;
      if (value) {
        const chunk = decoder.decode(value, { stream: true });
        // Pass false for isReplace so the dashboard appends the chunk
        onChunk(chunk, false);
      }
    }
  } catch (error) {
    console.warn(
      "Stream interrupted (likely backgrounded). Fetching full final answer...",
      error,
    );

    try {
      // Give the user a visual indicator that the app is fixing the connection
      onChunk("\n\n*(Connection paused. Retrieving final analysis...)*", false);

      // Trigger the standard fallback endpoint which will return the entire compiled answer
      const fallbackResponse = await fetch(`${BASE_URL}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query_text: userQuestion }),
      });

      if (!fallbackResponse.ok) throw new Error("Fallback request failed");

      const finalData = await fallbackResponse.json();
      const fullAnswer = finalData.response || finalData.answer;

      // Pass true for isReplace to completely overwrite the broken text with the final response
      onChunk(fullAnswer, true);
    } catch (fallbackError) {
      console.error("Streaming and fallback both failed:", fallbackError);
      onChunk(
        "\n\n❌ Network disconnected while you were away. Please ask again.",
        false,
      );
    }
  }
}
