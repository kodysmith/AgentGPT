import { Tool } from "langchain/tools";
import type { ModelSettings } from "../../utils/types";
import { LLMChain } from "langchain/chains";
import { createModel, summarizeSearchSnippets } from "../../utils/prompts";

/**
 * Wrapper around Google Search API adapted from Serper implementation in LangChain: https://github.com/hwchase17/langchainjs/blob/main/langchain/src/tools/serper.ts
 *
 * You can create a free API key at https://developers.google.com/custom-search/v1/overview.
 *
 * To use, you should have the GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX environment variables set.
 */
export class GoogleSearch extends Tool {
  // Required values for Tool
  name = "search";
  description =
    "A search engine that should be used sparingly and only for questions about current events. Input should be a search query.";

  protected apiKey: string;
  protected cx: string;
  protected modelSettings: ModelSettings;
  protected goal: string;

  constructor(modelSettings: ModelSettings, goal: string) {
    super();

    this.apiKey = process.env.GOOGLE_SEARCH_API_KEY ?? "";
    this.cx = process.env.GOOGLE_SEARCH_CX ?? "";
    this.modelSettings = modelSettings;
    this.goal = goal;
    if (!this.apiKey || !this.cx) {
      throw new Error(
        "Google Search API key or CX not set. You can set them as GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX in your .env file, or pass them to GoogleSearch."
      );
    }
  }

  /** @ignore */
  async _call(input: string) {
    const res = await this.callGoogleSearch(input);
    const searchResult: SearchResult = res.data;

    if (searchResult.items?.[0]?.pagemap?.metatags?.[0]['og:description']) {
      // Use Open Graph description if available
      return searchResult.items[0].pagemap.metatags[0]['og:description'];
    }

    if (searchResult.items?.[0]?.snippet) {
      const snippets = searchResult.items.map((result) => result.snippet);
      const summary = await summarizeSnippets(
        this.modelSettings,
        this.goal,
        input,
        snippets
      );
      const resultsToLink = searchResult.items.slice(0, 3);
      const links = resultsToLink.map((result) => result.link);

      return `${summary}\n\nLinks:\n${links
        .map((link) => `- ${link}`)
        .join("\n")}`;
    }

    return "No good search result found";
  }

  async callGoogleSearch(input: string) {
    const encodedQuery = encodeURIComponent(input);
    const options = {
      method: "GET",
    };

    const url = `https://www.googleapis.com/customsearch/v1?key=${this.apiKey}&cx=${this.cx}&q=${encodedQuery}`;
    const res = await fetch(url, options);

    if (!res.ok) {
      console.error(`Got ${res.status} error from Google Search API: ${res.statusText}`);
    }

    return res.json();
  }
}

interface SearchResult {
  items?: GoogleSearchResult[];
}

interface GoogleSearchResult {
  link: string;
  snippet: string;
  pagemap?: {
    metatags?: {
      'og:description'?: string;
    }[];
  };
}

const summarizeSnippets = async (
    modelSettings: ModelSettings,
    goal: string,
    query: string,
    snippets: string[]
  ) => {
    const prompt = `Summarize the following search results for the query "${query}":\n\n${snippets.join("\n\n")}`;
    const completion = await new LLMChain({
      llm: createModel(modelSettings),
      prompt,
    }).call({
      goal,
      query,
    });
    return completion.text as string;
  };