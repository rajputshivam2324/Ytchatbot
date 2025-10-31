import { fetchTranscript } from 'youtube-transcript-plus';
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import dotenv from 'dotenv';
dotenv.config({ path: '/home/shivam/ytchatbot/backend/.env' });
import { CloudClient } from "chromadb";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers"


const client = new CloudClient({
  apiKey: process.env.chroma,
  tenant: process.env.tenant,
  database: 'langchain',
});

const videoId = 'https://youtu.be/rWKwQ1I4xzc?si=rtEdEbwiEPxOBFm4';
const transcript = await fetchTranscript(videoId, { lang: 'en' });
    
const fullText = transcript.map(item => item.text).join(' ');


const textSplitter = new CharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
const chunks = await textSplitter.splitText(fullText);


const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "models/text-embedding-004",
  taskType: TaskType.RETRIEVAL_DOCUMENT,
});

const vectorStore = await Chroma.fromTexts(
  chunks,
  chunks.map((_, i) => ({ source: "youtube", chunkIndex: i })),
  embeddings,
  {
    collectionName: "ytchatbot",
    index: client as any, 
  }
);

const retriever = vectorStore.asRetriever();


const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash"
});

const question = "Explain complete summary of video in about 200 words";


const retrievedDocs = await retriever.invoke(question);


const contextText = retrievedDocs.map(doc => doc.pageContent).join("\n\n");

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system",`You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.`
  ],
  ["human", "Question: {question}\n\nContext:\n{context}"]
]);


const final_prompt = await promptTemplate.invoke({
  context: contextText,
  question
});

const parser= new StringOutputParser();
const chain = llm.pipe(parser)
const ans= await chain.invoke(final_prompt);
console.log(ans);

























