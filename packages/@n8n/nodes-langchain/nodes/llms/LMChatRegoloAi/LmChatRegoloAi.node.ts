import { ChatOpenAI, type ClientOptions } from '@langchain/openai';
import {
	NodeConnectionTypes,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
} from 'n8n-workflow';

import { getProxyAgent } from '@utils/httpProxyAgent';
import { getConnectionHintNoticeField } from '@utils/sharedFields';
import { openAiFailedAttemptHandler } from '../../vendors/OpenAi/helpers/error-handling';
import { makeN8nLlmFailedAttemptHandler } from '../n8nLlmFailedAttemptHandler';
import { N8nLlmTracing } from '../N8nLlmTracing';

export class LmChatRegoloAi implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Regolo AI Chat Model',
		name: 'lmChatRegoloAi',
		icon: { light: 'file:regoloAiLight.svg', dark: 'file:regoloAiLight.dark.svg' },
		group: ['transform'],
		version: 1,
		description: 'For advanced usage with an AI chain (LangChain provider for Regolo)',
		defaults: { name: 'Regolo AI Chat Model' },
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Language Models', 'Root Nodes'],
				'Language Models': ['Chat Models (Recommended)'],
			},
			resources: {
				primaryDocumentation: [{ url: 'https://docs.regolo.ai/' }],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiLanguageModel],
		outputNames: ['Model'],

		credentials: [{ name: 'regoloApi', required: true }],

		requestDefaults: {
			ignoreHttpStatusErrors: true,
			baseURL:
				'={{ $parameter.options?.baseURL?.split("/").slice(0,-1).join("/") || $credentials?.url?.split("/").slice(0,-1).join("/") || "https://api.regolo.ai" }}',
		},

		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiChain, NodeConnectionTypes.AiAgent]),

			// MODEL
			{
				displayName: 'Model',
				name: 'model',
				type: 'options',
				required: true,
				description: 'Regolo chat model',
				options: [
					{ name: 'Llama-3.3-70B-Instruct', value: 'Llama-3.3-70B-Instruct' },
					{ name: 'Llama-3.1-8B-Instruct', value: 'Llama-3.1-8B-Instruct' },
					{ name: 'Phi-4', value: 'Phi-4' },
					{ name: 'Qwen3-8B', value: 'Qwen3-8B' },
					{ name: 'deepSeek-v3-0324', value: 'deepSeek-v3-0324' },
					{ name: 'gemma-3-27b-it', value: 'gemma-3-27b-it' },
					{ name: 'gpt-oss-120b', value: 'gpt-oss-120b' },
					{ name: 'maestrale-chat-v0.4-beta', value: 'maestrale-chat-v0.4-beta' },
					// Vision:
					// { name: 'Qwen2.5-VL-32B-Instruct', value: 'Qwen2.5-VL-32B-Instruct' },
				],
				default: 'Llama-3.3-70B-Instruct',
			},

			// OPTIONS
			{
				displayName: 'Options',
				name: 'options',
				placeholder: 'Add Option',
				description: 'Additional options to add',
				type: 'collection',
				default: {},
				options: [
					{
						displayName: 'Base URL',
						name: 'baseURL',
						type: 'string',
						default: 'https://api.regolo.ai/v1',
						description: 'Override the default base URL for the Regolo API',
					},
					{
						displayName: 'Sampling Temperature',
						name: 'temperature',
						type: 'number',
						default: 0.7,
						typeOptions: { maxValue: 2, minValue: 0, numberPrecision: 1 },
						description: 'Controls randomness',
					},
					{
						displayName: 'Top P',
						name: 'topP',
						type: 'number',
						default: 1,
						typeOptions: { maxValue: 1, minValue: 0, numberPrecision: 1 },
						description: 'Nucleus sampling',
					},
					{
						displayName: 'Presence Penalty',
						name: 'presencePenalty',
						type: 'number',
						default: 0,
						typeOptions: { maxValue: 2, minValue: -2, numberPrecision: 1 },
					},
					{
						displayName: 'Frequency Penalty',
						name: 'frequencyPenalty',
						type: 'number',
						default: 0,
						typeOptions: { maxValue: 2, minValue: -2, numberPrecision: 1 },
					},
					{
						displayName: 'Maximum Number of Tokens',
						name: 'maxTokens',
						type: 'number',
						default: -1,
						typeOptions: { maxValue: 32768 },
					},
					{
						displayName: 'Response Format',
						name: 'responseFormat',
						type: 'options',
						default: 'text',
						options: [
							{ name: 'Text', value: 'text', description: 'Regular text response' },
							{ name: 'JSON', value: 'json_object', description: 'Force JSON output' },
						],
					},
					{
						displayName: 'Timeout',
						name: 'timeout',
						type: 'number',
						default: 60000,
						description: 'Max request time in ms',
					},
					{
						displayName: 'Max Retries',
						name: 'maxRetries',
						type: 'number',
						default: 2,
						description: 'Maximum number of retries to attempt',
					},
				],
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const credentials = await this.getCredentials('regoloApi');

		const modelName = this.getNodeParameter('model', itemIndex) as string;
		const options = this.getNodeParameter('options', itemIndex, {}) as {
			baseURL?: string;
			frequencyPenalty?: number;
			maxTokens?: number;
			maxRetries?: number;
			timeout?: number;
			presencePenalty?: number;
			temperature?: number;
			topP?: number;
			responseFormat?: 'text' | 'json_object';
		};

		const configuration: ClientOptions = {};

		if (options.baseURL) {
			configuration.baseURL = options.baseURL;
		} else if (credentials.url) {
			configuration.baseURL = credentials.url as string;
		}

		if (configuration.baseURL) {
			configuration.fetchOptions = {
				dispatcher: getProxyAgent(configuration.baseURL ?? 'https://api.regolo.ai/v1'),
			};
		}

		const modelKwargs: {
			response_format?: object;
		} = {};
		if (options.responseFormat) {
			modelKwargs.response_format = { type: options.responseFormat };
		}

		const model = new ChatOpenAI({
			apiKey: credentials.apiKey as string,
			model: modelName,
			temperature: options.temperature,
			topP: options.topP,
			presencePenalty: options.presencePenalty,
			frequencyPenalty: options.frequencyPenalty,
			maxTokens: options.maxTokens,
			timeout: options.timeout ?? 60000,
			maxRetries: options.maxRetries ?? 2,
			configuration,
			callbacks: [new N8nLlmTracing(this)],
			modelKwargs,
			onFailedAttempt: makeN8nLlmFailedAttemptHandler(this, openAiFailedAttemptHandler),
		});

		return { response: model };
	}
}
