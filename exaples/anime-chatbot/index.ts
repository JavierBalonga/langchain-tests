import dotenv from "dotenv";
import { LLMChain, OpenAI, PromptTemplate } from "langchain";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { FaissStore } from "langchain/vectorstores/faiss";
import path from "path";
import prompt from "prompt";

dotenv.config();

const {} = process.env;

const VECTOR_STORE_DIRECTORY = path.join(__dirname, "./data");

async function main() {
  // Vector store initialization
  let animeVectorStore: FaissStore;
  try {
    animeVectorStore = await FaissStore.load(
      VECTOR_STORE_DIRECTORY,
      new OpenAIEmbeddings()
    );
  } catch (error) {
    const animes: Anime[] = [];
    for (let i = 0; i < 4; i++) {
      const page = i + 1;
      console.log(`fetching page ${page} of 4`);

      const url = new URL("https://api.jikan.moe/v4/top/anime");
      url.searchParams.set("page", page.toString());
      url.searchParams.set("limit", String(25));
      const res = await fetch(url.toString());
      if (res.status >= 400) {
        console.error(res);
        throw new Error("Bad response from server");
      }
      const data: AnimeTopResponse = await res.json();

      animes.push(...data.data);

      // Jikan API rate limit is 4 requests per second
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    const texts: string[] = [];
    const textsMetadata: Anime[] = [];

    animes.forEach((anime) => {
      texts.push(`${anime.title}:\n${anime.synopsis}`);
      textsMetadata.push(anime);
    });

    animeVectorStore = await FaissStore.fromTexts(
      texts,
      textsMetadata,
      new OpenAIEmbeddings()
    );
    await animeVectorStore.save(VECTOR_STORE_DIRECTORY);
  }

  const LLModel = new OpenAI({
    modelName: "gpt-3.5-turbo",
    maxRetries: 1,
    temperature: 0,
  });

  const condenseChain = new LLMChain({
    llm: LLModel,
    prompt: PromptTemplate.fromTemplate(
      `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
      
      Chat History:
      {chat_history}
      Follow Up Input: {input}
      Standalone question:`
    ),
  });

  const answerChain = new LLMChain({
    llm: LLModel,
    prompt: PromptTemplate.fromTemplate(
      `You are an anime expert and your goal is provide to the user the best anime based in his requirements, provide a conversational answer to the user.
        You already have a list below with any possible anime that you can recommend to the user.
        But if you think that the user is not looking for any of the anime below just respond "I'm sorry, I'm not able to respond to your request".
      
        Here is the list of anime that you can recommend to the user:
        {anmeList}
      
        User:{querySearch}
        
        Answer in markdown:`
    ),
  });

  const messageHistory: { input: string; answer: string }[] = [];
  for (let i = 0; i < 10; i++) {
    const { input } = await prompt.get({
      properties: {
        input: {
          description: "User",
        },
      },
    });
    if (typeof input !== "string" || input === "exit") break;

    let condenseChainResponse: string;
    if (messageHistory.length === 0) {
      condenseChainResponse = input;
    } else {
      const res = await condenseChain.call({
        chat_history: messageHistory
          .map((message) => `User: ${message.input}\nBot: ${message.answer}`)
          .join("\n"),
        input,
      });
      condenseChainResponse = res.text as string;
    }

    const searchResult = await animeVectorStore.similaritySearch(
      condenseChainResponse,
      10
    );

    const { text: answerChainResponse } = await answerChain.call({
      querySearch: condenseChainResponse,
      anmeList: searchResult.map((anime) => anime.pageContent).join("\n"),
    });

    console.log("answer:", answerChainResponse);

    messageHistory.push({
      input,
      answer: answerChainResponse as string,
    });
  }
}

main().catch(console.error);

/* Inferred types from the api NO READ NEEDED */
export interface AnimeTopResponse {
  pagination: {
    last_visible_page: number;
    has_next_page: boolean;
    current_page: number;
    items: {
      count: number;
      total: number;
      per_page: number;
    };
  };
  data: Anime[];
}

export interface Anime {
  mal_id: number;
  url: string;
  images: { [key: string]: Image };
  trailer: Trailer;
  approved: boolean;
  titles: Title[];
  title: string;
  title_english: string;
  title_japanese: string;
  title_synonyms: string[];
  type: string;
  source: string;
  episodes: number;
  status: string;
  airing: boolean;
  aired: Aired;
  duration: string;
  rating: string;
  score: number;
  scored_by: number;
  rank: number;
  popularity: number;
  members: number;
  favorites: number;
  synopsis: string;
  background: null;
  season: string;
  year: number;
  broadcast: Broadcast;
  producers: Demographic[];
  licensors: Demographic[];
  studios: Demographic[];
  genres: Demographic[];
  explicit_genres: any[];
  themes: Demographic[];
  demographics: Demographic[];
}

export interface Aired {
  from: Date;
  to: Date;
  prop: Prop;
  string: string;
}

export interface Prop {
  from: From;
  to: From;
}

export interface From {
  day: number;
  month: number;
  year: number;
}

export interface Broadcast {
  day: string;
  time: string;
  timezone: string;
  string: string;
}

export interface Demographic {
  mal_id: number;
  type: Type;
  name: string;
  url: string;
}

export enum Type {
  Anime = "anime",
}

export interface Image {
  image_url: string;
  small_image_url: string;
  large_image_url: string;
}

export interface Title {
  type: string;
  title: string;
}

export interface Trailer {
  youtube_id: string;
  url: string;
  embed_url: string;
  images: Images;
}

export interface Images {
  image_url: string;
  small_image_url: string;
  medium_image_url: string;
  large_image_url: string;
  maximum_image_url: string;
}
