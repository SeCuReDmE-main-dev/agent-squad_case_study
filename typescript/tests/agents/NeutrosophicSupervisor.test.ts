import { BedrockLLMAgent, BedrockLLMAgentOptions } from "../../src/agents/bedrockLLMAgent";
import { NeutrosophicSupervisor } from "../../src/agents/neutrosophicSupervisor";
import { SupervisorAgentOptions } from "../../src/agents/supervisorAgent";
import { ConversationMessage, ParticipantRole } from "../../src/types";

class MockBedrockLLMAgent extends BedrockLLMAgent {
  constructor(options: BedrockLLMAgentOptions, private responseText = "Mock response") {
    super(options);
  }

  async processRequest(): Promise<ConversationMessage> {
    return {
      role: ParticipantRole.ASSISTANT,
      content: [{ text: this.responseText }],
    };
  }
}

function makeAgent(name: string, responseText: string): MockBedrockLLMAgent {
  return new MockBedrockLLMAgent(
    { name, description: `${name} description` },
    responseText
  );
}

function makeSupervisor(team: MockBedrockLLMAgent[]): NeutrosophicSupervisor {
  const leadAgent = makeAgent("Supervisor", "Lead response");
  return new NeutrosophicSupervisor({
    name: "SupervisorAgent",
    description: "My Supervisor agent description",
    leadAgent,
    team,
  } as SupervisorAgentOptions);
}

describe("NeutrosophicSupervisor", () => {
  test("adds consensus to agent responses", async () => {
    const supervisor = makeSupervisor([
      makeAgent("Billing Agent", "The answer is clear and complete for billing invoices."),
      makeAgent("Tech Agent", "Maybe this is unclear and could be a conflict."),
    ]);

    const response = await (supervisor as any).sendMessages([
      { recipient: "Billing Agent", content: "Help" },
      { recipient: "Tech Agent", content: "Help" },
    ]);

    expect(response).toContain("<neutrosophic_consensus>");
    expect(response).toContain("action:");
    expect(response).toContain("Billing Agent:");
    expect(response).toContain("Tech Agent:");
  });

  test("returns structured clarification when no agent matches", async () => {
    const supervisor = makeSupervisor([]);

    const response = await (supervisor as any).sendMessages([
      { recipient: "Unknown Agent", content: "Help" },
    ]);

    expect(response).toContain("<neutrosophic_consensus>");
    expect(response).toContain("action: CLARIFY");
    expect(response).toContain("I: 1.00");
  });

  test("rejects malformed messages", async () => {
    const supervisor = makeSupervisor([]);

    await expect((supervisor as any).sendMessages([{ recipient: "", content: "Help" }]))
      .rejects.toThrow("recipient must be a non-empty string");
    await expect((supervisor as any).sendMessages([{ recipient: "Agent", content: "" }]))
      .rejects.toThrow("content must be a non-empty string");
  });
});
