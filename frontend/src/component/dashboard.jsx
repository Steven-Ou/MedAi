import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import dynamic from "next/dynamic";
import "katex/dist/katex.min.css";
import {
  fetchDetectedPlants,
  triggerVisionScan,
  streamBotanicalQuestion,
  checkScanStatus,
  predictPlantImage,
} from "../utils/herbApi";

const ReactJoyride = dynamic(
  () => import("react-joyride").then((mod) => mod.default),
  {
    ssr: false,
    loading: () => null,
  },
);

export default function HerbAiDashboard() {
  const [telemetry, setTelemetry] = useState([]);
  const [isScanning, setIsScanning] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState("");
  const [runTour, setRunTour] = useState(true);
  const [videoSrc, setVideoSrc] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const videoRef = useRef(null);
  const [videoFile, setVideoFile] = useState(null);

  const tourSteps = [
    {
      target: ".media-upload-section", // You will need to add this className to your upload buttons div
      content:
        "Start here! Upload a video or image of a plant you want to identify.",
      placement: "bottom",
    },
    {
      target: ".identify-btn", // Add this className to your "Identify Footage" button
      content: "Click here to send your media to the YOLO vision model.",
    },
    {
      target: ".log-stream-container",
      content:
        "Once analyzed, all detected plants will appear here. Click on any row to load its clinical data!",
    },
  ];

  const handleVideoUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageSrc(null);
      setVideoSrc(URL.createObjectURL(file));
      setVideoFile(file);
    }
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoSrc(null);
      setImageSrc(URL.createObjectURL(file));
      setTelemetry([]);

      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          text: "Analyzing static photograph structural features...",
        },
      ]);
      try {
        const data = await predictPlantImage(file);
        if (data && data.predicted_class) {
          setTelemetry([
            {
              species: data.predicted_class,
              framesTracked: 1,
              maxConfidence: data.confidence,
              evidenceImage: URL.createObjectURL(file),
            },
          ]);

          setMessages((prev) => [
            ...prev,
            {
              role: "agent",
              text: `Inference Complete! Identified object as: **${data.predicted_class}** (Confidence: ${(data.confidence * 100).toFixed(0)}%). Click on the herb in the log stream to view its clinical profile.`,
            },
          ]);
        } else {
          throw new Error("Invalid response payload");
        }
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            text: "Failed reaching image-processing model ports.",
          },
        ]);
      }
    }
  };

  const handleStartScan = async () => {
    if (!videoFile) return alert("Please upload a video file first.");

    if (videoFile.size > 100 * 1024 * 1024)
      return alert("Video file is too large. Please keep it under 100MB.");

    setTelemetry([]);
    setIsScanning(true);
    if (videoRef.current) videoRef.current.play();

    setMessages((prev) => [
      ...prev,
      {
        role: "agent",
        text: "🎬 **Step 1: Video Analysis Initiated.**\n\nI am currently analyzing your video frame-by-frame. This is a heavy multimodal process, so **please be patient—you can leave this window open and grab a coffee**, I'll keep working in the background!\n\nOnce finished, the **Identification Log Stream** below will populate with every plant I detect. From there, you can click on any plant to trigger **Step 2**, where my Clinical RAG Agent will fetch detailed medicinal properties and you can ask follow-up questions.",
      },
    ]);

    try {
      const success = await triggerVisionScan(videoFile);
      if (!success) {
        setIsScanning(false);
        return;
      }

      const pollInterval = setInterval(async () => {
        const status = await checkScanStatus();

        try {
          const currentTelemetry = await fetchDetectedPlants();
          if (currentTelemetry) {
            setTelemetry(currentTelemetry.data || currentTelemetry);
          }
        } catch (err) {
          console.error(err);
        }

        if (!status.is_scanning) {
          setIsScanning(false);
          clearInterval(pollInterval);
          try {
            const finalTelemetry = await fetchDetectedPlants();
            const telemetryArray = finalTelemetry.data || finalTelemetry;

            if (telemetryArray && telemetryArray.length > 0) {
              setTelemetry(telemetryArray);
              const topPlant = telemetryArray.reduce((prev, current) =>
                prev.framesTracked > current.framesTracked ? prev : current,
              );
              setMessages((prev) => [
                ...prev,
                {
                  role: "agent",
                  text: `🎥 Video Inference Complete! I scanned the footage and predominantly identified: **${topPlant.species}** (Tracked across ${topPlant.framesTracked} frames). Click on it in the log stream below to view its details.`,
                },
              ]);
            }
          } catch (err) {
            console.error(err);
          }
        }
      }, 3000);
    } catch (err) {
      console.error(err);
      setIsScanning(false);
    }
  };

  const handleRowClick = async (speciesName) => {
    const autoQueryText = `Provide a structured clinical textbook profile for the medicinal substance: ${speciesName}. Include active compounds and biological properties.`;

    setMessages((prev) => [
      ...prev,
      { role: "user", text: `Tell me about ${speciesName}.` },
      { role: "agent", text: "", isTyping: true },
    ]);

    let streamedText = "";
    await streamBotanicalQuestion(autoQueryText, (chunk, isReplace = false) => {
      if (isReplace) {
        streamedText = chunk;
      } else {
        streamedText += chunk;
      }

      setMessages((prev) => {
        const newHistory = [...prev];
        const lastIndex = newHistory.length - 1;
        if (newHistory[lastIndex].role === "agent") {
          newHistory[lastIndex] = {
            ...newHistory[lastIndex],
            text: streamedText,
          };
        }
        return newHistory;
      });
    });

    setMessages((prev) => {
      const newHistory = [...prev];
      if (newHistory[newHistory.length - 1].role === "agent") {
        newHistory[newHistory.length - 1].isTyping = false;
      }
      return newHistory;
    });
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputQuery.trim()) return;

    const userMessageText = inputQuery;
    setInputQuery("");

    setMessages((prev) => [
      ...prev,
      { role: "user", text: userMessageText },
      { role: "agent", text: "", isTyping: true },
    ]);

    let streamedText = "";
    await streamBotanicalQuestion(
      userMessageText,
      (chunk, isReplace = false) => {
        if (isReplace) {
          streamedText = chunk;
        } else {
          streamedText += chunk;
        }

        setMessages((prev) => {
          const newHistory = [...prev];
          const lastIndex = newHistory.length - 1;
          if (newHistory[lastIndex].role === "agent") {
            newHistory[lastIndex] = {
              ...newHistory[lastIndex],
              text: streamedText,
            };
          }
          return newHistory;
        });
      },
    );

    setMessages((prev) => {
      const newHistory = [...prev];
      if (newHistory[newHistory.length - 1].role === "agent") {
        newHistory[newHistory.length - 1].isTyping = false;
      }
      return newHistory;
    });
  };

  const globalStyles = `
    html, body {
      margin: 0; padding: 0;
      background-color: #eaf4eb;

      background-image: 
        radial-gradient(circle at 15% 25%, rgba(212, 240, 208, 0.7) 0%, transparent 40%),
        radial-gradient(circle at 85% 75%, rgba(184, 226, 178, 0.6) 0%, transparent 45%),
        url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M25 25c15-15 30-7.5 37.5 7.5s-7.5 30-22.5 30-30-7.5-22.5-22.5 7.5-30 7.5-15zm-7.5 7.5c0 7.5 7.5 15 15 15M75 75c15-15 30-7.5 37.5 7.5s-7.5 30-22.5 30-30-7.5-22.5-22.5 7.5-30 7.5-15zm-7.5 7.5c0 7.5 7.5 15 15 15' fill='%236ea769' fill-opacity='0.15' stroke='%23488243' stroke-width='2' stroke-opacity='0.2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");

      background-attachment: fixed;
      background-size: auto, auto, 100px 100px;

      transition: background-color 0.4s ease; 
      height: 100%;
      overflow-x: hidden; 
    }
    
    .dashboard-wrapper {
      min-height: 100vh;
      padding: 25px 20px;
      box-sizing: border-box;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      width: 100%;
      overflow-x: hidden;
    }
    
    .dashboard-container {
      width: 100%;
      max-width: 1600px;
      margin: 0 auto;
    }
    
    .dashboard-header {
      background: linear-gradient(135deg, #065f46 0%, #0f766e 100%);
      padding: 25px 30px;
      border-radius: 20px;
      color: #fff;
      box-shadow: 0 8px 20px rgba(6, 95, 70, 0.15);
      margin-bottom: 25px;
      display: flex;
      flex-wrap: wrap;
      gap: 15px;
      align-items: center;
      justify-content: space-between;
    }

    .dashboard-grid {
      display: grid;
      grid-template-columns: minmax(350px, 1fr) minmax(450px, 1.5fr);
      gap: 25px;
      align-items: stretch; 
    }
    
    .panel-card {
      background-color: #ffffff;
      border-radius: 24px;
      padding: 30px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
      border: 1px solid #e2e8f0;
      min-height: calc(100vh - 150px);
      height: auto;
      display: flex;
      flex-direction: column;
      box-sizing: border-box;
      overflow-y: auto;
    }
    
    .log-stream-container {
      flex-grow: 1;
      overflow-y: auto;
      min-height: 150px;
      max-height: 250px; 
      border-radius: 8px;
    }
    
    .telemetry-row {
      cursor: pointer;
      transition: background-color 0.2s ease;
    }
    .telemetry-row:hover {
      background-color: #e2e8f0 !important;
    }

    /* --- STRICT CHAT BOUNDARIES --- */
    .chat-window {
      flex-grow: 1;
      overflow-y: auto;
      overflow-x: hidden;
      padding: 10px 10px 20px 10px;
      margin-bottom: 20px;
      width: 100%;
      box-sizing: border-box;
    }

    .chat-message-row {
      display: flex;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 24px;
      width: 100%;
      max-width: 100%;
      box-sizing: border-box;
    }
    
    .chat-message-row.user {
      flex-direction: row-reverse;
    }

    .chat-avatar {
      width: 38px;
      height: 38px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      flex-shrink: 0;
    }
    .chat-message-row.user .chat-avatar { background-color: #10b981; }
    .chat-message-row.agent .chat-avatar { background-color: #f1f5f9; }
    
    .msg-bubble {
      width: fit-content; 
      max-width: 85%; /* Limits bubble size on large screens */
      padding: 16px 20px;
      border-radius: 20px;
      font-size: 14.5px;
      color: #334155;
      
      white-space: normal; 
      word-wrap: break-word;
      overflow-wrap: break-word;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch; 
      box-sizing: border-box;
    }
    
    .msg-bubble > div {
       white-space: pre-wrap; /* Target the inner text wrapper for line breaks */
    }

    .msg-bubble.user {
      background-color: #d1fae5;
      border: 1px solid #a7f3d0;
      border-top-right-radius: 4px;
    }
    .msg-bubble.agent {
      background-color: #ffffff;
      border: 1px solid #e2e8f0;
      border-top-left-radius: 4px;
    }
    
    .markdown-body table { 
      display: block; 
      width: 100%; 
      max-width: 100%;
      overflow-x: auto; 
      -webkit-overflow-scrolling: touch;
      border-collapse: collapse; 
      margin: 15px 0; 
      white-space: normal; /* Prevents text from being forced into single unbreakable lines */
    }
    
    .markdown-body th, .markdown-body td { 
      min-width: 120px; /* Forces the table to be wide enough to trigger horizontal scrolling */
      border: 1px solid #e2e8f0; 
      padding: 10px; 
    }
    .markdown-body th { 
      background-color: #f8fafc; 
      color: #334155; 
    }
    
    @media (max-width: 1024px) {
      .dashboard-wrapper { padding: 10px 5px; }
      .dashboard-header { padding: 15px; }
      .dashboard-grid { 
        grid-template-columns: 1fr; 
        gap: 15px;
      }
      .panel-card { 
        height: auto; 
        min-height: 60vh;
        padding: 15px; 
      }
      .log-stream-container { 
        max-height: 300px; 
      }
      .msg-bubble { 
        padding: 12px 15px;
      }
      .chat-window {
        padding: 5px 5px 15px 5px;
      }
    }
    
    .panel-card::-webkit-scrollbar, .chat-window::-webkit-scrollbar, .log-stream-container::-webkit-scrollbar, .msg-bubble::-webkit-scrollbar { width: 6px; height: 6px; }
    .panel-card::-webkit-scrollbar-thumb, .chat-window::-webkit-scrollbar-thumb, .log-stream-container::-webkit-scrollbar-thumb, .msg-bubble::-webkit-scrollbar-thumb { 
        background-color: #cbd5e1; border-radius: 8px; 
    }
    
    @keyframes bounce {
      0%, 100% { transform: translateY(0); opacity: 0.5; }
      50% { transform: translateY(-3px); opacity: 1; }
    }
    .typing-dot { display: inline-block; animation: bounce 1.4s infinite ease-in-out both; margin: 0 1px; }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
  `;

  const styles = {
    viewport: {
      width: "100%",
      minHeight: "320px",
      backgroundColor: "#0f172a",
      borderRadius: "16px",
      overflow: "hidden",
      marginBottom: "20px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      border: "1px solid #334155",
      boxShadow: "inset 0 4px 12px rgba(0,0,0,0.5)",
    },
  };

  return (
    <div className="dashboard-wrapper">
      <ReactJoyride
        steps={tourSteps}
        run={runTour}
        continuous={true}
        showSkipButton={true}
        styles={{
          options: { primaryColor: "#10b981" },
        }}
      />
      <style dangerouslySetInnerHTML={{ __html: globalStyles }} />

      <div className="dashboard-container">
        <header className="dashboard-header">
          <div>
            <h1 style={{ margin: 0, fontSize: "26px", fontWeight: "700" }}>
              🌿 Herb-AI Systems Dashboard
            </h1>
            <p
              style={{ margin: "4px 0 0 0", opacity: 0.8, fontSize: "13.5px" }}
            >
              Vision Frameworks & RAG Clinical Intelligence
            </p>
          </div>
        </header>

        <div className="dashboard-grid">
          <div className="panel-card">
            <h3
              style={{
                fontSize: "17px",
                fontWeight: "600",
                color: "#065f46",
                margin: "0 0 15px 0",
              }}
            >
              📷 Media Upload Hub
            </h3>
            <div
              className="media-upload-section"
              style={{ display: "flex", gap: "10px", marginBottom: "20px" }}
            >
              {" "}
              <label
                style={{
                  flex: 1,
                  padding: "12px",
                  background: "#f8fafc",
                  border: "1px dashed #cbd5e1",
                  borderRadius: "10px",
                  textAlign: "center",
                  cursor: "pointer",
                  fontSize: "13.5px",
                }}
              >
                🎥 Upload Video
                <input
                  type="file"
                  accept="video/*,video/mp4,video/quicktime"
                  onChange={handleVideoUpload}
                  style={{ display: "none" }}
                />
              </label>
              <label
                style={{
                  flex: 1,
                  padding: "12px",
                  background: "#f8fafc",
                  border: "1px dashed #cbd5e1",
                  borderRadius: "10px",
                  textAlign: "center",
                  cursor: "pointer",
                  fontSize: "13.5px",
                }}
              >
                📸 Upload Herb Image
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  style={{ display: "none" }}
                />
              </label>
            </div>

            <button
              className="identify-btn"
              onClick={handleStartScan}
              style={{
                padding: "15px",
                background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                color: "#fff",
                border: "none",
                borderRadius: "14px",
                fontWeight: "600",
                marginBottom: "20px",
                cursor: "pointer",
              }}
            >
              {isScanning
                ? "🎥 Running Live Vector File Scanning Inference..."
                : "🚀 Identify Footage"}
            </button>
            <div style={styles.viewport}>
              {videoSrc && (
                <video
                  ref={videoRef}
                  src={videoSrc}
                  controls
                  muted
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                  }}
                />
              )}
              {imageSrc && (
                <img
                  src={imageSrc}
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                  }}
                  alt="Inference Target"
                />
              )}
            </div>
            <h3
              style={{
                fontSize: "17px",
                fontWeight: "600",
                color: "#065f46",
                margin: "15px 0 10px 0",
              }}
            >
              📊 Identification Log Stream
            </h3>

            <div className="log-stream-container">
              {telemetry.length === 0 ? (
                <div
                  style={{
                    textAlign: "center",
                    padding: "20px",
                    backgroundColor: "#f8fafc",
                    borderRadius: "8px",
                    border: "1px dashed #cbd5e1",
                  }}
                >
                  <p style={{ fontSize: "14px", color: "#64748b", margin: 0 }}>
                    Waiting for visual telemetry.
                    <br />
                    Upload media to populate logs.
                  </p>
                </div>
              ) : (
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "separate",
                    borderSpacing: "0 4px",
                  }}
                >
                  <tbody>
                    {telemetry.map((item, i) => (
                      <tr
                        key={i}
                        className="telemetry-row"
                        style={{ backgroundColor: "#f8fafc" }}
                        onClick={() => handleRowClick(item.species)}
                      >
                        <td style={{ padding: "8px", width: "50px" }}>
                          {item.evidenceImage ? (
                            <img
                              src={item.evidenceImage}
                              alt={item.species}
                              style={{
                                width: "45px",
                                height: "45px",
                                borderRadius: "8px",
                                objectFit: "cover",
                              }}
                            />
                          ) : (
                            <div
                              style={{
                                width: "45px",
                                height: "45px",
                                backgroundColor: "#e2e8f0",
                                borderRadius: "8px",
                              }}
                            />
                          )}
                        </td>
                        <td
                          style={{
                            padding: "12px 10px",
                            fontWeight: "600",
                            color: item.species.includes("Anomaly")
                              ? "#e74c3c"
                              : "#0f766e",
                          }}
                        >
                          {item.species}
                        </td>
                        <td
                          style={{
                            padding: "12px 10px",
                            color: "#64748b",
                            fontSize: "13.5px",
                          }}
                        >
                          {item.framesTracked} frames tracked
                        </td>
                        <td
                          style={{
                            padding: "12px 10px",
                            textAlign: "right",
                            fontWeight: "700",
                            color: "#10b981",
                          }}
                        >
                          {(item.maxConfidence * 100).toFixed(0)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          <div className="panel-card">
            <h3
              style={{
                fontSize: "17px",
                fontWeight: "600",
                color: "#065f46",
                margin: "0 0 15px 0",
              }}
            >
              💬 RAG Clinical Agent Terminal
            </h3>
            <div className="chat-window">
              {messages.length === 0 && (
                <div
                  style={{
                    textAlign: "center",
                    color: "#64748b",
                    marginTop: "10%",
                    padding: "0 20px",
                  }}
                >
                  <div style={{ fontSize: "40px", marginBottom: "15px" }}>
                    🌿
                  </div>
                  <p
                    style={{ margin: 0, fontSize: "14.5px", lineHeight: "1.6" }}
                  >
                    I am Herb-AI, your advanced, multimodal medical botanical
                    vision agent.
                    <br />
                    <br />
                    **Step 1:** Upload a video or image on the left, then click
                    "Identify Footage".
                    <br />
                    <br />
                    **Step 2:** Wait for the analysis to complete. The log
                    stream will populate with all detected plant life.
                    <br />
                    <br />
                    **Step 3:** Click on any detected herb in the log stream
                    below to instantly generate its structured clinical profile
                    and ask me specific follow-up questions!
                  </p>
                </div>
              )}
              {messages.map((msg, i) => (
                <div key={i} className={`chat-message-row ${msg.role}`}>
                  <div className="chat-avatar">
                    {msg.role === "user" ? "🧑‍🔬" : "🪴"}
                  </div>

                  <div className={`msg-bubble ${msg.role}`}>
                    {msg.role === "user" ? (
                      msg.text
                    ) : msg.isTyping && !msg.text ? (
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          color: "#0f766e",
                          fontWeight: 600,
                          fontSize: "18px",
                        }}
                      >
                        <span className="typing-dot">.</span>
                        <span className="typing-dot">.</span>
                        <span className="typing-dot">.</span>
                      </div>
                    ) : (
                      <div className="markdown-body">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                        >
                          {msg.text}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
            <form
              onSubmit={handleSendMessage}
              style={{ display: "flex", gap: "10px" }}
            >
              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                placeholder="Ask the agent to explain chemical uses or benefits..."
                style={{
                  flexGrow: 1,
                  padding: "16px",
                  borderRadius: "16px",
                  border: "1px solid #cbd5e1",
                  backgroundColor: "#f8fafc",
                  outline: "none",
                  fontSize: "16px",
                }}
              />
              <button
                type="submit"
                style={{
                  padding: "0 28px",
                  backgroundColor: "#065f46",
                  color: "#fff",
                  border: "none",
                  borderRadius: "16px",
                  fontWeight: "600",
                  cursor: "pointer",
                }}
              >
                Send
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
