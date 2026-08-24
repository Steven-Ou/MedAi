// frontend/src/component/dashboard.jsx
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
  const [bgColor, setBgColor] = useState("#f4f7f6"); // Cleaner background color
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
    if (!videoFile) {
      alert("Please upload a video file first.");
      return;
    }

    if (videoFile.size > 10 * 1024 * 1024) {
      alert(
        "Video file is too large. Please keep it under 10MB for the cloud pipeline.",
      );
      return;
    }

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
        console.error("The backend rejected the scan request.");
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
                  text: `🎥 Video Inference Complete! I scanned the footage and predominantly identified: **${topPlant.species}** (Tracked across ${topPlant.framesTracked} frames). Feel free to ask me to explain its clinical benefits below.`,
                },
              ]);
            } else {
              setMessages((prev) => [
                ...prev,
                {
                  role: "agent",
                  text: `🎥 Video Inference Complete! However, I couldn't confidently identify any specific herbs in the footage.`,
                },
              ]);
            }
          } catch (err) {
            console.error("Failed fetching post-scan telemetry:", err);
          }
        }
      }, 3000);
    } catch (err) {
      console.error("Failed reaching scan ports.", err);
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
      const lastIndex = newHistory.length - 1;
      if (newHistory[lastIndex].role === "agent") {
        newHistory[lastIndex].isTyping = false;
      }
      return newHistory;
    });
  };

  const globalStyles = `
    /* Enforced Side-by-Side Flex Layout */
    .dashboard-grid {
      display: grid;
      grid-template-columns: minmax(400px, 1fr) 1.5fr;
      gap: 25px;
      align-items: stretch; /* Forces equal height matching */
    }
    
    /* Dedicated internal scroll for each panel */
    .panel-card {
      height: calc(100vh - 150px);
      overflow-y: auto;
    }

    /* Wrap to stacked layout ONLY on mobile or small tablets */
    @media (max-width: 1024px) {
      .dashboard-grid {
        grid-template-columns: 1fr;
      }
      .panel-card {
        height: 60vh;
      }
    }

    /* Clean Scrollbars */
    .panel-card::-webkit-scrollbar, .chat-window::-webkit-scrollbar { width: 6px; }
    .panel-card::-webkit-scrollbar-thumb, .chat-window::-webkit-scrollbar-thumb { 
        background-color: #cbd5e1; 
        border-radius: 8px; 
    }

    .markdown-body table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    .markdown-body th, .markdown-body td { border: 1px solid #e2e8f0; padding: 10px; }
    .markdown-body th { background-color: #f8fafc; color: #334155; }
    .katex-display { overflow-x: auto; overflow-y: hidden; }

    @keyframes blink {
      0% { opacity: 0.2; transform: scale(0.8); }
      50% { opacity: 1; transform: scale(1.2); }
      100% { opacity: 0.2; transform: scale(0.8); }
    }
  `;

  const styles = {
    wrapper: {
      minHeight: "100vh",
      backgroundColor: bgColor,
      padding: "25px 20px",
      boxSizing: "border-box",
      fontFamily:
        '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      transition: "background-color 0.4s ease",
    },
    container: { width: "95vw", maxWidth: "1600px", margin: "0 auto" },
    header: {
      background: "linear-gradient(135deg, #065f46 0%, #0f766e 100%)", // Emerald Botanical Gradient
      padding: "25px 40px",
      borderRadius: "20px",
      color: "#fff",
      boxShadow: "0 8px 20px rgba(6, 95, 70, 0.15)",
      marginBottom: "25px",
      display: "flex",
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
      border: "1px solid #f1f5f9",
      borderRadius: "18px",
      padding: "20px",
      marginBottom: "20px",
      backgroundColor: "#fcfcfc",
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
                  fontWeight: "500",
                  color: "#334155",
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
                  fontWeight: "500",
                  color: "#334155",
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
                boxShadow: "0 4px 12px rgba(16, 185, 129, 0.3)",
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
              {!videoSrc && !imageSrc && (
                <div style={{ color: "#64748b", fontSize: "13.5px" }}>
                  Media Viewport Idle. Load an active file above.
                </div>
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
            <div style={{ overflowY: "auto", flexGrow: 1 }}>
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
                              src={`data:image/jpeg;base64,${item.evidenceImage}`}
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
                            color:
                              item.species.includes("Anomaly") ||
                              item.species.includes("Unidentified")
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
                    marginTop: "30%",
                    padding: "0 30px",
                  }}
                >
                  <div style={{ fontSize: "32px", marginBottom: "10px" }}>
                    🔬
                  </div>
                  Ask questions about medicine, herb properties, or check the
                  results of the video scan!
                </div>
              )}
              {messages.map((msg, i) => (
                <div
                  key={i}
                  style={{
                    marginBottom: "15px",
                    display: "flex",
                    justifyContent:
                      msg.role === "user" ? "flex-end" : "flex-start",
                  }}
                >
                  <div
                    style={{
                      padding: "12px 16px",
                      borderRadius: "12px",
                      backgroundColor:
                        msg.role === "user" ? "#ecfdf5" : "#f8fafc",
                      border:
                        msg.role === "user"
                          ? "1px solid #a7f3d0"
                          : "1px solid #e2e8f0",
                      color: "#334155",
                      maxWidth: "90%",
                      fontSize: "14.5px",
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "10px",
                        fontWeight: "700",
                        textTransform: "uppercase",
                        letterSpacing: "0.5px",
                        marginBottom: "6px",
                        opacity: 0.6,
                      }}
                    >
                      {msg.role === "user"
                        ? "Clinical Inquiry"
                        : "System Knowledge Matrix"}
                    </div>

                    {msg.role === "user" ? (
                      msg.text
                    ) : msg.isTyping && !msg.text ? (
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          color: "#0f766e",
                          fontWeight: 600,
                          gap: "6px",
                        }}
                      >
                        <span style={{ animation: "blink 1.4s infinite both" }}>
                          ●
                        </span>
                        <span
                          style={{
                            animation: "blink 1.4s infinite both",
                            animationDelay: "0.2s",
                          }}
                        >
                          ●
                        </span>
                        <span
                          style={{
                            animation: "blink 1.4s infinite both",
                            animationDelay: "0.4s",
                          }}
                        >
                          ●
                        </span>
                        <span style={{ marginLeft: "6px", fontSize: "13px" }}>
                          Synthesizing RAG response...
                        </span>
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
                  padding: "14px",
                  borderRadius: "12px",
                  border: "1px solid #cbd5e1",
                  outline: "none",
                  fontSize: "14px",
                }}
              />
              <button
                type="submit"
                style={{
                  padding: "0 24px",
                  backgroundColor: "#065f46",
                  color: "#fff",
                  border: "none",
                  borderRadius: "12px",
                  fontWeight: "600",
                  cursor: "pointer",
                  transition: "0.2s ease",
                }}
              >
                Query
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
