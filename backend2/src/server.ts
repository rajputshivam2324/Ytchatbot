import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { fetchTranscript } from "youtube-transcript-plus";

import { CharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";

import { CloudClient } from "chromadb";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { existsSync } from "fs";

// ----- ENV SETUP -----
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const envPath1 = join(__dirname, "../.env");
const envPath2 = join(__dirname, "../../.env");
dotenv.config({ path: existsSync(envPath1) ? envPath1 : envPath2 });

const app = express();
app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: "50mb" }));

// ----- SAFE VIDEO ID EXTRACTOR -----
function extractVideoId(url) {
  try {
    const u = new URL(url);
    if (u.searchParams.get("v")) return u.searchParams.get("v");
  } catch (_) {}

  const patterns = [
    /youtu\.be\/([^?]+)/,
    /embed\/([^?]+)/,
    /v=([^&]+)/,
  ];

  for (const p of patterns) {
    const m = url.match(p);
    if (m) return m[1];
  }

  throw new Error("Invalid YouTube URL. Cannot extract videoId.");
}

// --------------------------------------------------
// ---------------------- API ------------------------
// --------------------------------------------------

app.post("/ytchatbot", async (req, res) => {
  try {
    const { videoUrl, question, followUp } = req.body;

    if (!videoUrl || !question)
      return res.status(400).json({ error: "videoUrl and question are required" });

    const videoId = extractVideoId(videoUrl);
    const collectionName = `yt_${videoId}`;

    console.log("\n🎥 Video:", videoUrl);
    console.log("📁 Collection:", collectionName);
    console.log("🔄 Follow-up:", followUp);

    // -------- Initialize Chroma + Embeddings --------
    const client = new CloudClient({
      apiKey: process.env.chroma,
      tenant: process.env.tenant,
      database: "langchain",
    });

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      model: "models/text-embedding-004",
      taskType: TaskType.RETRIEVAL_DOCUMENT,
    });

    let vectorStore;

    // ---------------------------------------------------
    // -------- IF FOLLOW-UP: LOAD EXISTING DATA ---------
    // ---------------------------------------------------
    if (followUp === true) {
      console.log("🟡 Follow-up request → loading existing collection");

      try {
        vectorStore = await Chroma.fromExistingCollection(embeddings, {
          collectionName,
          index: client,
        });
      } catch (err) {
        return res.status(404).json({
          error: "No previous context found for follow-up question.",
        });
      }
    }

    // ---------------------------------------------------
    // -------- NEW VIDEO: FETCH + SAVE TRANSCRIPT -------
    // ---------------------------------------------------
    else {
      console.log("🟢 New video → resetting collection");

      // Delete any old data
      try {
        await client.deleteCollection({ name: collectionName });
      } catch (_) {}

      // ---- FIXED TRANSCRIPT CALL ----
      console.log("⏳ Fetching transcript...");
      const transcript = await fetchTranscript(videoId); // FIXED

      if (!transcript || transcript.length === 0) {
        console.log("❌ Transcript not found");
        return res.status(400).json({
          error: "Transcript unavailable. Check if the video has captions.",
        });
      }

      const fullText = transcript.map(t => t.text).join(" ");

      // ---- Split into chunks ----
      const splitter = new CharacterTextSplitter({
        chunkSize: 1200,
        chunkOverlap: 200,
      });

      const chunks = await splitter.splitText(fullText);
      console.log(`📝 Created ${chunks.length} transcript chunks`);

      const ids = chunks.map((_, i) => `${videoId}_${Date.now()}_${i}`);

      // ---- Store in Chroma ----
      vectorStore = await Chroma.fromTexts(
        chunks,
        ids.map((id, i) => ({ id, videoId, index: i })),
        embeddings,
        { collectionName, index: client }
      );
    }

    // Small delay for Chroma sync
    await new Promise(r => setTimeout(r, 300));

    // ----------------------------
    // -------- RETRIEVAL ---------
    // ----------------------------
    const retriever = vectorStore.asRetriever();

    console.log("🔍 Retrieving context for question:", question);
    const docs = await retriever.getRelevantDocuments(question);
    const context = docs.map(d => d.pageContent).join("\n\n");

    // ----------------------------
    // ---------- LLM -------------
    // ----------------------------
    const llm = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash",
      apiKey: process.env.GOOGLE_API_KEY,
      temperature: 0.3,
    });

    const prompt = `
Answer ONLY using the transcript chunks provided.
If the answer is not available, reply: "Not available in transcript."

QUESTION:
${question}

CONTEXT:
${context}
`;

    console.log("💬 Generating response...");
    const answer = await llm.invoke(prompt);

    return res.json({ answer });

  } catch (err) {
    console.error("🔥 ERROR:", err);
    res.status(500).json({ error: "Internal Error", message: err.message });
  }
});

// --------------------------------------------------
// ------------------- SERVER -----------------------
// --------------------------------------------------
const PORT = process.env.PORT || 3005;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
