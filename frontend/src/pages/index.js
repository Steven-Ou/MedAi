import React from "react";
import Head from "next/head";
import HerbAiDashboard from "../component/dashboard"; // Corrected path and casing

function App() {
  return (
    <>
      <Head>
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"
        />
        <title>Herb-AI Dashboard</title>
      </Head>
      <div className="App">
        <HerbAiDashboard />
      </div>
    </>
  );
}

export default App;