import dotenv from "dotenv";
import prompt from "prompt";
import { Calculator } from "langchain/tools/calculator";
import { OpenAI } from "langchain/llms/openai";
import { SerpAPI } from "langchain/tools";
import { initializeAgentExecutorWithOptions } from "langchain/agents";

dotenv.config();

const { SERPAPI_API_KEY } = process.env;

async function main() {
  prompt.start();

  const { input } = await prompt.get({
    properties: {
      input: {
        default:
          "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?",
      },
    },
  });

  if (typeof input !== "string") {
    throw new Error("input must be a string");
  }

  const executor = await initializeAgentExecutorWithOptions(
    // Tools
    [
      new SerpAPI(SERPAPI_API_KEY, {
        location: "Austin,Texas,United States",
        hl: "en",
        gl: "us",
      }),
      new Calculator(),
    ],
    // LLM model
    new OpenAI({ temperature: 0 }),
    {
      agentType: "zero-shot-react-description",
      verbose: true,
    }
  );

  const result = await executor.call({ input });

  console.log(result);
}

main().catch(console.error);
