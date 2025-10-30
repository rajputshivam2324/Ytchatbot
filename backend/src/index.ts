import { fetchTranscript } from 'youtube-transcript-plus';
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
import { MongoClient } from "mongodb";
import dotenv from 'dotenv';
dotenv.config({ path: '/home/shivam/ytchatbot/backend/.env' });


const videoId = 'https://youtu.be/rWKwQ1I4xzc?si=rtEdEbwiEPxOBFm4';
const transcript = await fetchTranscript(videoId,{lang:'en'});
console.log('Transcript fetched successfully:');
const fullText = transcript.map(item => item.text).join(' ');
console.log(fullText);

const textSplitter = new CharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
const chunks = await textSplitter.splitText(fullText);
console.log(chunks);

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "models/text-embedding-004", // This is the latest model
  taskType: TaskType.RETRIEVAL_DOCUMENT, // or e.g., TaskType.SEMANTIC_SIMILARITY
});

const uri = process.env.MONGODB_ATLAS_URI;
const dbName = process.env.MONGODB_ATLAS_DB_NAME;
const collectionName = process.env.MONGODB_ATLAS_COLLECTION_NAME;


const client = new MongoClient(uri);
const collection = client
  .db(dbName)
  .collection(collectionName);


const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
  collection: collection,
  indexName: "vector_index",
  embeddingKey: "embedding",
});

await client.connect();

const documents = chunks.map((chunk, idx) => ({
  pageContent: chunk,
  metadata: { videoId, chunkIndex: idx }
})) as any[];

await vectorStore.addDocuments(documents as any);

console.log("Embeddings stored in MongoDB successfully.");

await client.close();