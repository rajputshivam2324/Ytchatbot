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

// Determine .env path
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const envPath1 = join(__dirname, "../.env");
const envPath2 = join(__dirname, "../../.env");
const envPath = existsSync(envPath1) ? envPath1 : envPath2;
dotenv.config({ path: envPath });

const app = express();
app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: "50mb" }));

// ---------------------------------------------------------------
// HELPER — Extract Video ID
// ---------------------------------------------------------------
function extractVideoId(url) {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
    /youtube\.com\/watch\?.*v=([^&\n?#]+)/
  ];

  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match && match[1]) return match[1];
  }

  return Buffer.from(url)
    .toString("base64")
    .replace(/[^a-zA-Z0-9]/g, "")
    .substring(0, 16);
}

// ---------------------------------------------------------------
//  MAIN ENDPOINT
// ---------------------------------------------------------------
app.post("/ytchatbot", async (req, res) => {
  try {
    const { videoUrl, question, followUp } = req.body;

    if (!videoUrl || !question)
      return res.status(400).json({ error: "videoUrl and question are required" });

    const videoId = extractVideoId(videoUrl);
    const collectionName = `yt_${videoId}`;

    console.log("\n🔗 Video:", videoUrl);
    console.log("🎯 Collection:", collectionName);
    console.log("↩ Follow-up:", followUp);

    // -----------------------------------------------------------
    // Chroma Setup
    // -----------------------------------------------------------
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

    // -----------------------------------------------------------
    // FOLLOW-UP ⇒ Load existing collection
    // -----------------------------------------------------------
    if (followUp === true) {
      console.log("🟡 Loading existing collection (follow-up mode)");
      vectorStore = await Chroma.fromExistingCollection(embeddings, {
        collectionName,
        index: client,
      });
    }

    // -----------------------------------------------------------
    // NEW VIDEO ⇒ Delete & Recreate collection
    // -----------------------------------------------------------
    else {
      console.log("🟥 NEW video → Resetting collection");

      try {
        await client.deleteCollection({ name: collectionName });
      } catch (e) {
        console.log("ℹ No previous collection to delete");
      }

      console.log("⏳ Fetching transcript...");
      const transcript = await fetchTranscript(videoUrl, { lang: "en" });

      if (!transcript || transcript.length === 0)
        return res.status(400).json({
          error: "No transcript found. Enable captions on the video.",
        });

      const fullText = transcript.map((t) => t.text).join(" ");

      const splitter = new CharacterTextSplitter({
        chunkSize: 1200,
        chunkOverlap: 200,
      });

      const chunks = await splitter.splitText(fullText);

      // Unique IDs for each chunk
      const ids = chunks.map((_, i) => `${videoId}_${Date.now()}_${i}`);

      console.log(`🧩 Creating ${chunks.length} chunks...`);

      vectorStore = await Chroma.fromTexts(
        chunks,
        ids.map((id, i) => ({ id, videoId, index: i })),
        embeddings,
        { collectionName, index: client }
      );

      // Small wait = avoids “first request fail”
      await new Promise((r) => setTimeout(r, 400));
    }

    // -----------------------------------------------------------
    // RAG: Retriever + LLM
    // -----------------------------------------------------------
    const retriever = vectorStore.asRetriever();

    const llm = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash",
      apiKey: process.env.GOOGLE_API_KEY,
      temperature: 0.3,
    });

    console.log("🔍 Retrieving context...");
    const docs = await retriever.getRelevantDocuments(question);
    const context = docs.map((d) => d.pageContent).join("\n\n");

    const prompt = `
Answer ONLY using the transcript.
If the answer is not present, say: "Not available in transcript."

QUESTION:
${question}

CONTEXT:
${context}
    `;

    console.log("🤖 Generating answer...");
    const answer = await llm.invoke(prompt);

    return res.json({ answer });
  } catch (err) {
    console.error("🔥 ERROR:", err);
    res.status(500).json({
      error: "Internal Server Error",
      message: err.message,
    });
  }
});

// ---------------------------------------------------------------
// START SERVER
// ---------------------------------------------------------------
const PORT = process.env.PORT || 3005;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
