import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
  import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
  import {
    ChatPromptTemplate,
    MessagesPlaceholder,
  } from "@langchain/core/prompts";
  import { StateGraph } from "@langchain/langgraph";
  import { Annotation } from "@langchain/langgraph";
  import { tool } from "@langchain/core/tools";
  import { ToolNode } from "@langchain/langgraph/prebuilt";
//   import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
  import { MemorySaver } from "@langchain/langgraph";
  import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
  import { MongoClient } from "mongodb";
  import { z } from "zod";
  import "dotenv/config";
  
  // Ensure you have GOOGLE_API_KEY in your .env or environment
  // or use service account for Vertex AI if in GCP
  
  export async function callAgent(client: MongoClient, query: string, thread_id: string) {
    // Define the MongoDB database and collection
    const dbName = "hr_database";
    const db = client.db(dbName);
    const collection = db.collection("employees");
  
    // Define the graph state
    const GraphState = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
      }),
    });
  
    // Define the tools for the agent to use
    const employeeLookupTool = tool(
      async ({ query, n = 10 }) => {
        console.log("Employee lookup tool called");
  
        const dbConfig = {
          collection: collection,
          indexName: "vector_index",
          textKey: "embedding_text",
          embeddingKey: "embedding",
        };
  
        // Use Google's Vertex AI Embeddings
        const vectorStore = new MongoDBAtlasVectorSearch(
            new GoogleGenerativeAIEmbeddings({
              model: "embedding-001", // or "text-embedding-004" if available
              apiKey: process.env.GOOGLE_API_KEY, // pass API key here too
            }),
            dbConfig
          );
  
        const result = await vectorStore.similaritySearchWithScore(query, n);
        return JSON.stringify(result);
      },
      {
        name: "employee_lookup",
        description: "Gathers employee details from the HR database",
        schema: z.object({
          query: z.string().describe("The search query"),
          n: z
            .number()
            .optional()
            .default(10)
            .describe("Number of results to return"),
        }),
      }
    );
  
    const tools = [employeeLookupTool];
    const toolNode = new ToolNode<typeof GraphState.State>(tools);
  
    // Use Google's Gemini model instead of Anthropic
    const model = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-pro", // or gemini-1.5-flash
      temperature: 0,
      apiKey: process.env.GOOGLE_API_KEY, // or auto-loaded from env
    }).bindTools(tools);
  
    // Determine next step: tool call or end
    function shouldContinue(state: typeof GraphState.State) {
      const messages = state.messages;
      const lastMessage = messages[messages.length - 1] as AIMessage;
  
      if (lastMessage.tool_calls?.length) {
        return "tools";
      }
      return "__end__";
    }
  
    // Model calling function
    async function callModel(state: typeof GraphState.State) {
      const prompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          `You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress towards answering the question. If you are unable to fully answer, that's OK, another assistant with different tools will help where you left off. Execute what you can to make progress. If you or any of the other assistants have the final answer or deliverable, prefix your response with FINAL ANSWER so the team knows to stop. You have access to the following tools: {tool_names}.\n{system_message}\nCurrent time: {time}.`,
        ],
        new MessagesPlaceholder("messages"),
      ]);
  
      const formattedPrompt = await prompt.formatMessages({
        system_message: "You are a helpful HR Chatbot Agent.",
        time: new Date().toISOString(),
        tool_names: tools.map((tool) => tool.name).join(", "),
        messages: state.messages,
      });
  
      const result = await model.invoke(formattedPrompt);
      return { messages: [result] };
    }
  
    // Build the graph
    const workflow = new StateGraph(GraphState)
      .addNode("agent", callModel)
      .addNode("tools", toolNode)
      .addEdge("__start__", "agent")
      .addConditionalEdges("agent", shouldContinue)
      .addEdge("tools", "agent");
  
    // Use MongoDB as checkpointer
    // const checkpointer = new MongoDBSaver({ client, dbName });
    const checkpointer = new MemorySaver();
    const app = workflow.compile({ checkpointer });
  
    // Invoke the agent
    const finalState = await app.invoke(
      {
        messages: [new HumanMessage(query)],
      },
      { recursionLimit: 15, configurable: { thread_id: thread_id } }
    );
  
    const finalMessage = finalState.messages[finalState.messages.length - 1];
    console.log(finalMessage.content);
    return finalMessage.content;
  }