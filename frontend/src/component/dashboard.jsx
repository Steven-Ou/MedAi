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

const SHOW_SYSTEM_NOTICE = true;

// 1. THE CORRECT NAMED EXPORT
const ReactJoyride = dynamic(
  () =>
    import("react-joyride").then((mod) => {
      const JoyrideComponent =
        mod.default?.default || mod.default || mod.Joyride || mod;

      return function JoyrideSafeWrapper(props) {
        return <JoyrideComponent {...props} />;
      };
    }),
  { ssr: false },
);



// 2. THE MASCOT TOOLTIP (Safely outside the main function)
const MascotTooltip = ({
  index,
  step,
  backProps,
  primaryProps,
  skipProps,
  tooltipProps,
  isLastStep,
}) => (
  <div
    {...tooltipProps}
    style={{
      display: "flex",
      alignItems: "center",
      backgroundColor: "#1e293b",
      padding: "20px",
      borderRadius: "16px",
      color: "#f8fafc",
      maxWidth: "450px",
      gap: "15px",
      boxShadow: "0 10px 25px rgba(0,0,0,0.2)",
    }}
  >
    <div style={{ flexShrink: 0 }}>
      <img
        src="/mascot.png"
        alt="Agent Mascot"
        style={{ width: "80px", height: "80px", imageRendering: "pixelated" }}
      />
    </div>
    <div style={{ flexGrow: 1 }}>
      <div
        style={{
          fontSize: "17px",
          marginBottom: "15px",
          lineHeight: "1.5",
          fontFamily: "'VT323', monospace",
        }}
      >
        {step.content}
      </div>
      <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px" }}>
        {/* ADDED: The Skip Button */}
        <button
          {...skipProps}
          style={{
            backgroundColor: "transparent",
            color: "#94a3b8",
            border: "none",
            cursor: "pointer",
            fontWeight: "bold",
            fontFamily: "'VT323', monospace",
            fontSize: "18px",
          }}
        >
          Skip
        </button>

        {index > 0 && (
          <button
            {...backProps}
            style={{
              backgroundColor: "transparent",
              color: "#cbd5e1",
              border: "none",
              cursor: "pointer",
              fontWeight: "bold",
              fontFamily: "'VT323', monospace",
              fontSize: "18px",
            }}
          >
            Back
          </button>
        )}
        <button
          {...primaryProps}
          style={{
            backgroundColor: "#10b981",
            color: "white",
            padding: "8px 16px",
            borderRadius: "8px",
            border: "none",
            cursor: "pointer",
            fontWeight: "bold",
            fontFamily: "'VT323', monospace",
            fontSize: "18px",
          }}
        >
          {isLastStep ? "Got it!" : "Next"}
        </button>
      </div>
    </div>
  </div>
);

export default function HerbAiDashboard() {
  const [isMounted, setIsMounted] = useState(false);
  const [telemetry, setTelemetry] = useState([]);
  const [isScanning, setIsScanning] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState("");

  // 3. TOUR STATES
  const [showNotice, setShowNotice] = useState(SHOW_SYSTEM_NOTICE);

  const [runTour, setRunTour] = useState(false);
  const [tourKey, setTourKey] = useState(0);
  const [isInitialTour, setIsInitialTour] = useState(true);

  const [videoSrc, setVideoSrc] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const videoRef = useRef(null);
  const [videoFile, setVideoFile] = useState(null);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // 4. TOUR STEPS WITH BEACONS DISABLED
  const baseSteps = [
    {
      target: "body", 
      placement: "center",
      content: "Welcome to Herb-AI! Let me show you around the dashboard.",
      disableBeacon: true, 
    },
    {
      target: ".media-upload-section",
      content: "Start here! Upload a video or image of a plant you want to identify.",
      placement: "right",
      disableBeacon: true,
    },
    {
      target: ".identify-btn",
      content: "Click here to send your media to the YOLO vision model.",
      placement: "bottom",
      disableBeacon: true,
    },
    {
      target: ".log-stream-container", 
      content: "Once analyzed, all detected plants will appear here. Click on any row to load its clinical data!",
      placement: "top-start",
      disableBeacon: true,
    },
    {
      target: ".chat-terminal-section", 
      content: "This is the RAG Clinical Agent Terminal! Here, you can ask the AI follow-up questions about the identified herbs.",
      placement: "left",
      disableBeacon: true,
    },
  ];

  // Appends the outro ONLY on the first run
  const tourSteps = isInitialTour
    ? [
        ...baseSteps,
        {
          target: ".tour-trigger-btn",
          content: "You're all set! If you ever need a refresher, just click here to retake the tour.",
          placement: "bottom",
          disableBeacon: true,
        },
      ]
    : baseSteps;
  // 5. UNCONTROLLED CALLBACK
  const handleJoyrideCallback = (data) => {
    const { status } = data;
    if (["finished", "skipped"].includes(status)) {
      setRunTour(false);
      setIsInitialTour(false);
    }
  };

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
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

    html, body {
      margin: 0; padding: 0;
      background-color: #ecfdf5; /* Soft mythical jade background */
      /* 8-Bit Gold & Jade Grid Background */
      background-image: 
        linear-gradient(rgba(167, 243, 208, 0.6) 2px, transparent 2px),
        linear-gradient(90deg, rgba(167, 243, 208, 0.6) 2px, transparent 2px),
        radial-gradient(circle at 15% 25%, rgba(212, 240, 208, 0.7) 0%, transparent 40%),
        radial-gradient(circle at 85% 75%, rgba(184, 226, 178, 0.6) 0%, transparent 45%),
        url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M25 25c15-15 30-7.5 37.5 7.5s-7.5 30-22.5 30-30-7.5-22.5-22.5 7.5-30 7.5-15zm-7.5 7.5c0 7.5 7.5 15 15 15M75 75c15-15 30-7.5 37.5 7.5s-7.5 30-22.5 30-30-7.5-22.5-22.5 7.5-30 7.5-15zm-7.5 7.5c0 7.5 7.5 15 15 15' fill='%236ea769' fill-opacity='0.15' stroke='%23488243' stroke-width='2' stroke-opacity='0.2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
      
      background-size: 32px 32px, 32px 32px, auto, auto, 100px 100px;
      background-attachment: fixed;
      font-family: 'VT323', monospace;
      height: 100%;
      overflow-x: hidden;
    }
    
    .dashboard-wrapper {
      min-height: 100vh; padding: 25px 20px; box-sizing: border-box;
      font-family: 'VT323', monospace;
      width: 100%; overflow-x: hidden;
      font-size: 24px; 
    }
    .dashboard-container { width: 100%; max-width: 1600px; margin: 0 auto; }
    
    .dashboard-header {
      background: #064e3b; 
      padding: 25px 30px; 
      border-radius: 8px; 
      color: #a7f3d0; /* Soft mint text */
      border: 4px solid #047857; /* Deep emerald border instead of gold */
      box-shadow: 6px 6px 0px #047857; /* Deep emerald shadow */
      margin-bottom: 35px;
      display: flex; flex-wrap: wrap; gap: 15px; align-items: center; justify-content: space-between;
    }

    .dashboard-grid { display: grid; grid-template-columns: minmax(350px, 1fr) minmax(450px, 1.5fr); gap: 35px; align-items: stretch; }
    
    .panel-card {
      background-color: #ffffff; 
      border-radius: 4px; 
      padding: 30px;
      border: 4px solid #064e3b;
      box-shadow: 8px 8px 0px #064e3b;
      min-height: calc(100vh - 150px); height: auto; display: flex; flex-direction: column;
      box-sizing: border-box; 
      overflow-y: auto; 
    }
    
    .log-stream-container { flex-grow: 1; overflow: visible; border-radius: 0px; border: 4px solid #a7f3d0; padding: 10px; }
    .telemetry-row { cursor: pointer; transition: background-color 0.1s; }
    .telemetry-row:hover { background-color: #ecfdf5 !important; }
    
    .chat-window { flex-grow: 1; overflow-y: auto; overflow-x: hidden; padding: 10px 10px 20px 10px; margin-bottom: 20px; width: 100%; box-sizing: border-box; }
    .chat-message-row { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 24px; width: 100%; max-width: 100%; box-sizing: border-box; }
    .chat-message-row.user { flex-direction: row-reverse; }
    .chat-avatar { width: 44px; height: 44px; border: 4px solid #064e3b; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 24px; flex-shrink: 0; background-color: #fef08a; }
    
    .msg-bubble {
      width: fit-content; max-width: 85%; padding: 16px 20px; 
      border-radius: 4px; font-size: 24px;
      color: #064e3b; white-space: normal; word-wrap: break-word; overflow-wrap: break-word; overflow-x: auto;
      border: 4px solid #064e3b;
      box-shadow: 4px 4px 0px #064e3b;
      box-sizing: border-box;
    }
    .msg-bubble > div { white-space: pre-wrap; }
    .msg-bubble.user { background-color: #a7f3d0; }
    .msg-bubble.agent { background-color: #ffffff; }
    
    .markdown-body table { display: block; width: 100%; max-width: 100%; overflow-x: auto; border-collapse: collapse; margin: 15px 0; white-space: normal; }
    .markdown-body th, .markdown-body td { min-width: 120px; border: 4px solid #064e3b; padding: 10px; }
    .markdown-body th { background-color: #f8fafc; color: #064e3b; }
    
    /* Interactive Button Press Animation */
    .pixel-btn { transition: transform 0.1s, box-shadow 0.1s; }
    .pixel-btn:active { transform: translate(4px, 4px) !important; box-shadow: 0px 0px 0px #064e3b !important; }
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
      {isMounted && (
        <ReactJoyride
          key={tourKey}
          steps={tourSteps}
          run={runTour}
          continuous={true}
          showSkipButton={true}
          tooltipComponent={MascotTooltip}
          callback={handleJoyrideCallback}
          disableScrollParentFix={true} // Prevents clipping!
          styles={{
            options: {
              zIndex: 10000,
              primaryColor: "#064e3b",
            },
          }}
        />
      )}
      <style dangerouslySetInnerHTML={{ __html: globalStyles }} />

      <div className="dashboard-container">
        {SHOW_SYSTEM_NOTICE && showNotice && (
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100vw",
              height: "100vh",
              backgroundColor: "rgba(6, 95, 70, 0.85)",
              zIndex: 999999,
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              backdropFilter: "blur(8px)",
            }}
          >
            <div
              style={{
                backgroundColor: "#ffffff",
                borderRadius: "24px",
                padding: "40px",
                maxWidth: "600px",
                width: "90%",
                textAlign: "center",
                border: "4px solid #10b981",
                boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
              }}
            >
              <div style={{ fontSize: "60px", marginBottom: "20px" }}>🚧</div>
              <h2
                style={{
                  color: "#065f46",
                  fontSize: "28px",
                  margin: "0 0 15px 0",
                }}
              >
                Development Warning
              </h2>
              <p
                style={{
                  color: "#475569",
                  fontSize: "16px",
                  lineHeight: "1.6",
                  marginBottom: "30px",
                }}
              >
                This project is currently undergoing live updates. You may
                encounter temporary errors or models failing to respond while
                architecture changes are pushed.
                <br />
                <br />
                If you encounter any issues, please reach out via my contact
                email in the footer.
              </p>
              <button
                onClick={() => {
                  setShowNotice(false);
                  setRunTour(true);
                }}
                style={{
                  backgroundColor: "#10b981",
                  color: "#fff",
                  padding: "14px 32px",
                  borderRadius: "12px",
                  border: "none",
                  fontSize: "18px",
                  fontWeight: "bold",
                  cursor: "pointer",
                  boxShadow: "0 4px 10px rgba(16, 185, 129, 0.3)",
                }}
              >
                I Understand, Continue
              </button>
            </div>
          </div>
        )}

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
          <button
            className="tour-trigger-btn pixel-btn"
            onClick={() => {
              setIsInitialTour(false);
              setTourKey((prev) => prev + 1);
              setRunTour(true);
            }}
            style={{
              backgroundColor: "#10b981",
              color: "white",
              padding: "10px 20px",
              borderRadius: "8px",
              border: "4px solid #1e293b",
              cursor: "pointer",
              fontWeight: "600",
              boxShadow: "4px 4px 0px #1e293b",
              fontFamily: "'VT323', monospace",
              fontSize: "22px",
            }}
          >
            🗺️ Take a Tour
          </button>
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
              className="identify-btn pixel-btn"
              onClick={handleStartScan}
              style={{
                padding: "15px",
                background: "#10b981",
                color: "#fff",
                border: "4px solid #1e293b",
                borderRadius: "8px",
                fontWeight: "600",
                marginBottom: "20px",
                cursor: "pointer",
                boxShadow: "6px 6px 0px #1e293b",
                fontFamily: "'VT323', monospace",
                fontSize: "24px",
              }}
            >
              {isScanning ? "🎥 Scanning Media..." : "🚀 Identify Footage"}
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

            <div
              className="log-section"
              style={{
                minHeight: "180px",
                padding: "10px",
                backgroundColor: "#ffffff",
                borderRadius: "16px",
                position: "relative",
                zIndex: 1,
                overflow: "visible"
              }}
            >
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
                    <p
                      style={{ fontSize: "14px", color: "#64748b", margin: 0 }}
                    >
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
          </div>

          <div className="panel-card chat-terminal-section">
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
                placeholder="Ask the agent..."
                style={{
                  flexGrow: 1,
                  padding: "16px",
                  borderRadius: "8px",
                  border: "4px solid #1e293b",
                  backgroundColor: "#f8fafc",
                  outline: "none",
                  fontSize: "22px",
                  fontFamily: "'VT323', monospace",
                }}
              />
              <button
                type="submit"
                className="pixel-btn"
                style={{
                  padding: "0 28px",
                  backgroundColor: "#065f46",
                  color: "#fff",
                  border: "4px solid #1e293b",
                  borderRadius: "8px",
                  fontWeight: "600",
                  cursor: "pointer",
                  boxShadow: "4px 4px 0px #1e293b",
                  fontFamily: "'VT323', monospace",
                  fontSize: "22px",
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
