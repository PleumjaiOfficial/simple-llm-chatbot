import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from "@supabase/supabase-js"

// load and check path in local
import dotenv from 'dotenv';
import path from 'path';
dotenv.config();

// variable for api key
const openAIApiKey = process.env.OPENAI_API_KEY
const sbApiKey = process.env.SUPABASE_API_KEY
const sbUrl = process.env.SUPABASE_URL

// const of vectorstore
const embeddings = new OpenAIEmbeddings({openAIApiKey})
const client = createClient(sbUrl, sbApiKey)

const vectorstores = new SupabaseVectorStore(embeddings, {
    client,
    tableName: 'documents',
    queryName: 'match_documents'
})

// source: https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/
const retriever = vectorstores.asRetriever()

export {retriever}
