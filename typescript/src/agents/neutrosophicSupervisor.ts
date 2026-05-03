import { Agent } from "./agent";
import { SupervisorAgent, SupervisorAgentOptions } from "./supervisorAgent";
import {
  decide,
  neutrosophicEvidenceConsensus,
  scoreTextResponse,
  Triplet,
} from "../neutrosophic";

type SupervisorMessage = { recipient: string; content: string };
type ScoredResponse = { agentName: string; responseText: string; score: Triplet };

export class NeutrosophicSupervisor extends SupervisorAgent {
  constructor(options: SupervisorAgentOptions) {
    super(options);
  }

  protected async sendMessages(messages: SupervisorMessage[]): Promise<string> {
    this.validateMessages(messages);

    const tasks = messages
      .map((message) => {
        const agent = this.team.find((candidate) => candidate.name === message.recipient);
        return agent ? this.sendScoredMessage(agent, message.content) : null;
      })
      .filter((task): task is Promise<ScoredResponse> => task !== null);

    if (!tasks.length) {
      const consensus = new Triplet(0.0, 1.0, 0.0);
      return this.formatNeutrosophicResponse([], consensus, decide(consensus));
    }

    const scoredResponses = await Promise.all(tasks);
    const consensus = neutrosophicEvidenceConsensus(
      scoredResponses.map((response) => response.score)
    );

    return this.formatNeutrosophicResponse(
      scoredResponses,
      consensus,
      decide(consensus)
    );
  }

  private async sendScoredMessage(agent: Agent, content: string): Promise<ScoredResponse> {
    const rawResponse = await this.sendMessage(
      agent,
      content,
      this.userId,
      this.sessionId,
      this.additionalParams
    );
    const prefix = `${agent.name}: `;
    const responseText = rawResponse.startsWith(prefix)
      ? rawResponse.slice(prefix.length)
      : rawResponse;

    return {
      agentName: agent.name,
      responseText,
      score: scoreTextResponse(responseText),
    };
  }

  private validateMessages(messages: SupervisorMessage[]): void {
    if (!Array.isArray(messages)) {
      throw new TypeError("messages must be an array");
    }

    messages.forEach((message, index) => {
      if (typeof message !== "object" || message === null) {
        throw new TypeError(`messages[${index}] must be an object`);
      }
      if (typeof message.recipient !== "string" || !message.recipient.trim()) {
        throw new Error(`messages[${index}].recipient must be a non-empty string`);
      }
      if (typeof message.content !== "string" || !message.content.trim()) {
        throw new Error(`messages[${index}].content must be a non-empty string`);
      }
    });
  }

  private formatNeutrosophicResponse(
    scoredResponses: ScoredResponse[],
    consensus: Triplet,
    action: string
  ): string {
    const scoreLines = scoredResponses.map((response) => (
      `- ${response.agentName}: ${response.responseText}\n` +
      `  T=${response.score.T.toFixed(2)}, I=${response.score.I.toFixed(2)}, F=${response.score.F.toFixed(2)}`
    ));

    return (
      "<neutrosophic_consensus>\n" +
      `action: ${action}\n` +
      `T: ${consensus.T.toFixed(2)}\n` +
      `I: ${consensus.I.toFixed(2)}\n` +
      `F: ${consensus.F.toFixed(2)}\n` +
      "</neutrosophic_consensus>\n" +
      "<agent_responses>\n" +
      scoreLines.join("\n") +
      "\n</agent_responses>"
    );
  }
}
