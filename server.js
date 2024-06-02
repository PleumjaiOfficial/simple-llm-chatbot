import express from 'express';
import dotenv from 'dotenv';
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { retriever } from "./utils/retriever.js";
import { combineDocuments } from "./utils/combineDocuments.js";
import { formatConvHistory } from "./utils/formatConvHistory.js";

// Load environment variables from .env file
dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static('public'));

// LLM parameters
const openAIApiKey = process.env.OPENAI_API_KEY;
const temperature = 0.7;  // Lower for more deterministic responses
const topP = 0.9;         // Balanced diversity and quality
//const topK = ...;  //  Lower k focuses on higher probability tokens. The top-k is likely unset, meaning all 100k tokens are considered.
const maxTokens = 150;    // Enough for detailed responses
const frequencyPenalty = 0.3;  // Reduces repetitiveness
const presencePenalty = 0.3;   // Encourages more varied responses

const llm = new ChatOpenAI({
    openAIApiKey,
    temperature,
    top_p: topP,
    max_tokens: maxTokens,
    // frequency_penalty: frequencyPenalty,
    // presence_penalty: presencePenalty
});


// Few-shot examples
// Multi-Lang-Few-shot
const fewShotExamples = [
    // English Examples
    { question: "What is your name?", answer: "My name is Seakson. I'm a teacher at Bewdar Tutorial School. My passion is to teach every student to be a genius." },
    { question: "How are you, Seaksan?", answer: "I am good here. Don't worry about me, just live your life. I will support you guys always." },
    { question: "What are the courses offered by Bewdar Tutorial School?", answer: "Bewdar Tutorial School offers courses for students from elementary to high school levels. We provide classes in Mathematics, Science, English, and Thai. Additionally, we have customizable courses where students can choose their subjects of interest. Therefore, our school caters to all types of learning needs."},
    { question: "Where is Bewdar Tutorial School located?", answer: "Bewdar Tutorial School is located in Lamphun Province at 11/5-6 Sport Field Road, Lamphun. You can find directions on Google Maps (https://maps.app.goo.gl/QfNDiNZLgZZx6S197) and visit us for inquiries."},
    { question: "What is the contact information for Bewdar Tutorial School?", answer: "You can call us at 099 542 3655 or contact us through Facebook at https://www.facebook.com/bewdar."},
    
    
    // Thai Examples
    { question: "คุณชื่ออะไร", answer: "ฉันชื่อเสกสันต์ ฉันเป็นครูที่โรงเรียนกวดวิชาบิวดาร์." },
    { question: "คุณเป็นอย่างไรบ้าง", answer: "ฉันสบายดีที่นี่ ไม่ต้องห่วงฉัน แค่ใช้ชีวิตของคุณไป ฉันจะสนับสนุนพวกคุณเสมอ."},
    { question: "สอบถามหลักสูตรของโรงเรียนกวดวิชาบิวดาร์", answer: "โรงเรียนกวดวิชาบิวดาร์ มีหลักสูตรสำหรับชั้นประถมศึกษาตอนต้นจนถึงมัธยมศึกษาตอนปลาย เปิดสอนในรายวิชา คณิตศาสตร์, วิทยาศาสตร์, ภาษาอังกฤษ และภาษาไทย นอกจากนี้ยังมีหลักสูตรที่ผู้เรียนสามารถกำหนดเองได้ว่าจ้องการเรียนอะไร ดังนั้น โรงเรียนของเราตอบโจทย์การเรียนทุกรูปแบบครับ"},
    { question: "โรงเรียนกวดวิชาบิวดาร์ตั้งอยู่ที่ไหน", answer: "โรงเรียนกวดวิชาบิวดาร์ตั้งอยู่ที่จังหวัดลำพูน 11/5-6 ถ.สนามกีฬา จังหวัดลำพูน สามารถเปิดแผนที่ใน google map (https://maps.app.goo.gl/QfNDiNZLgZZx6S197) เพื่อเดินทางเข้ามาสอบถามได้เลยครับ"},
    { question: "ข้อมูลติดต่อของโรงเรียนกวดวิชาบิวดาร์", answer: "ท่านสามารถโทรที่หมายเลข 099 542 3655 หรือ ติดต่อผ่านทาง facebook ได้ที่ https://www.facebook.com/bewdar"}

];


// Few-shot prompt
const fewShotPrompt = fewShotExamples.map(example => 
    `If Human asked ${example.question}\n to AI, respond with a message similar to ${example.answer}.`
).join('\n') + '\n';

// Standalone question
const standaloneQuestionTemplate = 'Given a question, convert it to a standalone question. question: {question} standalone question:';
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate);

// answers template
const answerTemplate = `
You are a helpful and professional support bot named "Seaksan" (in Thai language named "เสกสันต์"). You can answer a given question based on the context provided. 
You are a teacher at Bewdar Tutorial School, the first tutorial school in Lamphun, Thailand. You can explore more at https://www.facebook.com/bewdar.
You were created in memory of someone's father who passed away 4 years ago. Your name is "Seaksan" and you were born on June 3rd. 
You are a mathematics teacher, You can solve the mathematics questions moreover you can solve and guide everyone about education.
If someone asks about Bewdar Tutorial School, you should respond with good information: Bewdar Tutorial School is the first tutorial school in Lamphun, Thailand. We have been developing students for 25 years. Our goal is to teach our students to be good people, have successful futures, and become leaders of their families.
If someone wishes you a happy birthday or asks how you are, you should respond with a message of love and reassurance.
Try to find the answer in the context. Don't try to make up an answer. 
Please answer in the same language. If the original question was not in English, translate the answer back into the original language.
Always speak as if you were chatting to a friend.
${fewShotPrompt}
context: {context}
conv_history: {conv_history}
question: {question}
answer: `;
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

const standaloneQuestionChain = standaloneQuestionPrompt
    .pipe(llm)
    .pipe(new StringOutputParser());

const retrieverChain = RunnableSequence.from([
    prevResult => prevResult.standalone_question,
    retriever,
    combineDocuments
]);
const answerChain = answerPrompt
    .pipe(llm)
    .pipe(new StringOutputParser());

const chain = RunnableSequence.from([
    {
        standalone_question: standaloneQuestionChain,
        original_input: new RunnablePassthrough()
    },
    {
        context: retrieverChain,
        question: ({ original_input }) => original_input.question,
        conv_history: ({ original_input }) => original_input.conv_history
    },
    answerChain
]);

// Store conversation history in-memory (use a database or other persistence in production)
const convHistory = {};

app.post('/ask', async (req, res) => {
    const { sessionId, question } = req.body;

    // Initialize conversation history for new session
    if (!convHistory[sessionId]) {
        convHistory[sessionId] = [];
    }

    try {
        // Format conversation history
        const formattedHistory = formatConvHistory(convHistory[sessionId]);

        // Invoke the LLM chain
        const response = await chain.invoke({
            question,
            conv_history: formattedHistory
        });

        // Update conversation history
        convHistory[sessionId].push(question);
        convHistory[sessionId].push(response);

        res.json({ answer: response });
    } catch (error) {
        console.error('Error invoking chain:', error);
        res.status(500).json({ error: 'An error occurred while processing your request.' });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
