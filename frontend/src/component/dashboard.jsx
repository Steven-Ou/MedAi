import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import {
  fetchDetectedPlants,
  triggerVisionScan,
  streamBotanicalQuestion,
  checkScanStatus,
  uploadImage,
} from "../utils/herbApi";

export default function HerbAiDashboard() {
  const [telemetry, setTelemetry] = useState([]);
  const [isScanning, setIsScanning] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState("");

  const [apiOnline, setApiOnline] = useState(false);
  const [bgColor, setBgColor] = useState("#f4f7f6");
  const [videoSrc, setVideoSrc] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const videoRef = useRef(null);
  const [videoFile, setVideoFile] = useState(null);

  const fetchTelemetry = async () => {
    try {
      const data = await fetchDetectedPlants();
      if (data) {
        setTelemetry(data.data || data);
        setApiOnline(true);
      } else {
        setApiOnline(false);
      }
    } catch (err) {
      console.error("Failed fetching database telemetry strings:", err);
      setApiOnline(false);
    }
  };

  useEffect(() => {
    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 4000);
    return () => clearInterval(interval);
  }, []);

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

      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          text: "Analyzing static photograph structural features...",
        },
      ]);
      try {
        const data = await uploadImage(file);
        if (data && data.predicted_class) {
          setMessages((prev) => [
            ...prev,
            {
              role: "agent",
              text: `Inference Complete! Identified object as: **${data.predicted_class}** (Confidence: ${(data.confidence * 100).toFixed(0)}%). Feel free to ask me to explain its clinical benefits below.`,
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
    if (videoFile.size > 10 * 1024 * 1024)
      return alert("Video file is too large.");

    setIsScanning(true);
    if (videoRef.current) videoRef.current.play();

    setMessages((prev) => [
      ...prev,
      {
        role: "agent",
        text: "🎬 Initiating video stream telemetry scan... Please wait while I extract and analyze the frames.",
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
        if (!status.is_scanning) {
          setIsScanning(false);
          clearInterval(pollInterval);
          try {
            const telemetryData = await fetchDetectedPlants();
            if (telemetryData && telemetryData.length > 0) {
              const topPlant = telemetryData.reduce((prev, current) =>
                prev.framesTracked > current.framesTracked ? prev : current,
              );
              setMessages((prev) => [
                ...prev,
                {
                  role: "agent",
                  text: `🎥 Video Inference Complete! I scanned the footage and predominantly identified: **${topPlant.species}** (Tracked across ${topPlant.framesTracked} frames).`,
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
    await streamBotanicalQuestion(userMessageText, (chunk) => {
      streamedText += chunk;
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

  const globalStyles = `
    html, body {
      margin: 0; padding: 0;
      background-color: ${bgColor};
      transition: background-color 0.4s ease; height: 100%;
    }
    .dashboard-grid {
      display: grid;
      /* SWAPPED: Media Hub on Left (1fr), Chat on Right (1.5fr) */
      grid-template-columns: minmax(350px, 1fr) minmax(450px, 1.5fr);
      gap: 25px;
      align-items: stretch; 
    }
    .panel-card {
      height: calc(100vh - 150px);
      display: flex;
      flex-direction: column;
    }
    .log-stream-container {
      flex-grow: 1;
      overflow-y: auto;
      min-height: 150px;
      max-height: 250px; 
      border-radius: 8px;
    }
    
    /* MOBILE SQUEEZE FIX */
    @media (max-width: 1024px) {
      .dashboard-grid { 
        grid-template-columns: 1fr; 
      }
      .panel-card { 
        height: auto; /* Allows content to push the container height naturally */
        min-height: 60vh;
      }
      .log-stream-container { 
        max-height: 300px; 
      }
    }
    
    .panel-card::-webkit-scrollbar, .chat-window::-webkit-scrollbar, .log-stream-container::-webkit-scrollbar { width: 6px; }
    .panel-card::-webkit-scrollbar-thumb, .chat-window::-webkit-scrollbar-thumb, .log-stream-container::-webkit-scrollbar-thumb { 
        background-color: #cbd5e1; border-radius: 8px; 
    }
    .markdown-body table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    .markdown-body th, .markdown-body td { border: 1px solid #e2e8f0; padding: 10px; }
    .markdown-body th { background-color: #f8fafc; color: #334155; }
    @keyframes bounce {
      0%, 100% { transform: translateY(0); opacity: 0.5; }
      50% { transform: translateY(-3px); opacity: 1; }
    }
    .typing-dot { display: inline-block; animation: bounce 1.4s infinite ease-in-out both; margin: 0 1px; }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
  `;

  const styles = {
    wrapper: {
      minHeight: "100vh",
      padding: "25px 20px",
      boxSizing: "border-box",
      fontFamily:
        '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    },
    container: { width: "95vw", maxWidth: "1600px", margin: "0 auto" },
    header: {
      background: "linear-gradient(135deg, #065f46 0%, #0f766e 100%)",
      padding: "25px 30px", // slightly reduced padding for mobile
      borderRadius: "20px",
      color: "#fff",
      boxShadow: "0 8px 20px rgba(6, 95, 70, 0.15)",
      marginBottom: "25px",
      display: "flex",
      flexWrap: "wrap",    // FIX: Allows elements to stack on mobile
      gap: "15px",         // FIX: Adds space between stacked elements
      alignItems: "center",
      justifyContent: "space-between",
    },
    panelCardStyles: {
      backgroundColor: "#ffffff",
      borderRadius: "24px",
      padding: "30px",
      boxShadow: "0 10px 30px rgba(0, 0, 0, 0.04)",
      border: "1px solid #e2e8f0",
      display: "flex",
      flexDirection: "column",
    },
    themeSelector: {
      display: "flex",
      gap: "8px",
      alignItems: "center",
      backgroundColor: "rgba(255,255,255,0.15)",
      padding: "6px 12px",
      borderRadius: "12px",
    },
    themeBtn: {
      width: "20px",
      height: "20px",
      borderRadius: "50%",
      border: "2px solid #fff",
      cursor: "pointer",
    },
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
    chatWindow: {
      flexGrow: 1,
      overflowY: "auto",
      padding: "10px 20px 20px 10px",
      marginBottom: "20px",
    },
  };

  return (
    <div style={styles.wrapper}>
      <style>{globalStyles}</style>
      <div style={styles.container}>
        <header style={styles.header}>
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
          <div
            style={{
              backgroundColor: apiOnline
                ? "rgba(255, 255, 255, 0.9)"
                : "rgba(211, 47, 47, 0.9)",
              padding: "8px 16px",
              borderRadius: "30px",
              color: apiOnline ? "#065f46" : "#fff",
              fontSize: "13px",
              fontWeight: "700",
            }}
          >
            {apiOnline ? "🟢 CORE API: ONLINE" : "🔴 BACKEND DISCONNECTED"}
          </div>
          <div style={styles.themeSelector}>
            <span
              style={{
                fontSize: "12px",
                marginRight: "4px",
                fontWeight: "600",
              }}
            >
              🎨 Theme:
            </span>
            <div
              onClick={() => setBgColor("#f4f7f6")}
              style={{ ...styles.themeBtn, backgroundColor: "#f4f7f6" }}
            />
            <div
              onClick={() => setBgColor("#e8f5e9")}
              style={{ ...styles.themeBtn, backgroundColor: "#e8f5e9" }}
            />
            <div
              onClick={() => setBgColor("#fef9e7")}
              style={{ ...styles.themeBtn, backgroundColor: "#fef9e7" }}
            />
            <div
              onClick={() => setBgColor("#1e293b")}
              style={{ ...styles.themeBtn, backgroundColor: "#1e293b" }}
            />
          </div>
        </header>

        <div className="dashboard-grid">
          {/* ========================================= */}
          {/* MEDIA UPLOAD HUB (MOVED TO LEFT) */}
          {/* ========================================= */}
          <div className="panel-card" style={styles.panelCardStyles}>
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
            <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
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
                🎥 Load Video Walk
                <input
                  type="file"
                  accept="video/*"
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
                : "🚀 Execute Pipeline Stream Scan"}
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
                <p style={{ fontSize: "13.5px", color: "#94a3b8" }}>
                  No logs committed to tracking schemas yet.
                </p>
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
                      <tr key={i} style={{ backgroundColor: "#f8fafc" }}>
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
          {/* ========================================= */}

          {/* ========================================= */}
          {/* TERMINAL PANEL (MOVED TO RIGHT) */}
          {/* ========================================= */}
          <div className="panel-card" style={styles.panelCardStyles}>
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
            <div className="chat-window" style={styles.chatWindow}>
              {messages.length === 0 && (
                <div
                  style={{
                    textAlign: "center",
                    color: "#94a3b8",
                    marginTop: "20%",
                    padding: "0 30px",
                  }}
                >
                  <div style={{ fontSize: "40px", marginBottom: "15px" }}>
                    🌿
                  </div>
                  <h4 style={{ color: "#475569", margin: "0 0 10px 0" }}>
                    Ready to assist!
                  </h4>
                  <p style={{ margin: 0, fontSize: "14.5px" }}>
                    Ask questions about medicine, herb properties, or check the
                    results of the video scan.
                  </p>
                </div>
              )}
              {messages.map((msg, i) => (
                <div
                  key={i}
                  style={{
                    marginBottom: "24px",
                    display: "flex",
                    flexDirection: msg.role === "user" ? "row-reverse" : "row",
                    alignItems: "flex-start",
                    gap: "12px",
                  }}
                >
                  <div
                    style={{
                      width: "38px",
                      height: "38px",
                      borderRadius: "50%",
                      backgroundColor:
                        msg.role === "user" ? "#10b981" : "#f1f5f9",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "20px",
                      flexShrink: 0,
                    }}
                  >
                    {msg.role === "user" ? "🧑‍🔬" : "🪴"}
                  </div>
                  <div
                    style={{
                      padding: "16px 20px",
                      borderRadius: "20px",
                      borderTopRightRadius:
                        msg.role === "user" ? "4px" : "20px",
                      borderTopLeftRadius:
                        msg.role === "agent" ? "4px" : "20px",
                      backgroundColor:
                        msg.role === "user" ? "#d1fae5" : "#ffffff",
                      border:
                        msg.role === "user"
                          ? "1px solid #a7f3d0"
                          : "1px solid #e2e8f0",
                      color: "#334155",
                      maxWidth: "85%",
                      fontSize: "14.5px",
                      whiteSpace: "pre-wrap",
                    }}
                  >
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
                  fontSize: "14.5px",
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
          {/* ========================================= */}
        </div>
      </div>
    </div>
  );
}
