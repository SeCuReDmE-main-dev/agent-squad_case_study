from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions, NeutrosophicSupervisor, SupervisorAgent
from agent_squad.agents.supervisor_agent import SupervisorAgentOptions
from agent_squad.storage import InMemoryChatStorage
from agent_squad.types import ConversationMessage, ParticipantRole


class MockBedrockLLMAgent(BedrockLLMAgent):
    def __init__(self, options, response_text="Mock response"):
        super().__init__(options)
        self.response_text = response_text

    async def process_request(self, *args, **kwargs):
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": self.response_text}]
        )


@pytest.fixture
def mock_boto3_client():
    with patch('boto3.client') as mock_client:
        yield mock_client


def mock_storage():
    storage = MagicMock(spec=InMemoryChatStorage)
    storage.save_chat_messages = AsyncMock()
    storage.fetch_chat = AsyncMock(return_value=[])
    storage.fetch_all_chats = AsyncMock(return_value=[])
    return storage


def make_agent(name, response_text):
    return MockBedrockLLMAgent(
        BedrockLLMAgentOptions(name=name, description=f"{name} description"),
        response_text=response_text,
    )


@pytest.mark.asyncio
async def test_neutrosophic_supervisor_adds_consensus_to_agent_responses(mock_boto3_client):
    lead_agent = make_agent("Supervisor", "Lead response")
    team = [
        make_agent("Billing Agent", "The answer is clear and complete for billing invoices."),
        make_agent("Tech Agent", "Maybe this is unclear and could be a conflict."),
    ]
    supervisor = NeutrosophicSupervisor(SupervisorAgentOptions(
        name="SupervisorAgent",
        description="My Supervisor agent description",
        lead_agent=lead_agent,
        team=team,
        storage=mock_storage(),
    ))

    response = await supervisor.send_messages([
        {"recipient": "Billing Agent", "content": "Help"},
        {"recipient": "Tech Agent", "content": "Help"},
    ])

    assert "<neutrosophic_consensus>" in response
    assert "action:" in response
    assert "T:" in response
    assert "I:" in response
    assert "F:" in response
    assert "Billing Agent:" in response
    assert "Tech Agent:" in response


@pytest.mark.asyncio
async def test_neutrosophic_supervisor_raises_indeterminacy_for_conflicting_evidence(mock_boto3_client):
    lead_agent = make_agent("Supervisor", "Lead response")
    team = [
        make_agent("Clear Agent", "The task completed successfully with a clear answer."),
        make_agent("Error Agent", "The task failed with an invalid response and an error."),
    ]
    supervisor = NeutrosophicSupervisor(SupervisorAgentOptions(
        name="SupervisorAgent",
        description="My Supervisor agent description",
        lead_agent=lead_agent,
        team=team,
        storage=mock_storage(),
    ))

    response = await supervisor.send_messages([
        {"recipient": "Clear Agent", "content": "Help"},
        {"recipient": "Error Agent", "content": "Help"},
    ])

    assert "action: CLARIFY" in response
    assert "Error Agent:" in response


@pytest.mark.asyncio
async def test_neutrosophic_supervisor_returns_structured_no_match_response(mock_boto3_client):
    lead_agent = make_agent("Supervisor", "Lead response")
    supervisor = NeutrosophicSupervisor(SupervisorAgentOptions(
        name="SupervisorAgent",
        description="My Supervisor agent description",
        lead_agent=lead_agent,
        team=[],
        storage=mock_storage(),
    ))

    response = await supervisor.send_messages([
        {"recipient": "Unknown Agent", "content": "Help"},
    ])

    assert "<neutrosophic_consensus>" in response
    assert "action: CLARIFY" in response
    assert "T: 0.00" in response
    assert "I: 1.00" in response
    assert "<agent_responses>" in response


@pytest.mark.asyncio
async def test_neutrosophic_supervisor_rejects_invalid_messages(mock_boto3_client):
    lead_agent = make_agent("Supervisor", "Lead response")
    supervisor = NeutrosophicSupervisor(SupervisorAgentOptions(
        name="SupervisorAgent",
        description="My Supervisor agent description",
        lead_agent=lead_agent,
        team=[],
        storage=mock_storage(),
    ))

    with pytest.raises(ValueError, match="recipient must be a non-empty string"):
        await supervisor.send_messages([{"recipient": "", "content": "Help"}])

    with pytest.raises(ValueError, match="content must be a non-empty string"):
        await supervisor.send_messages([{"recipient": "Agent", "content": ""}])


@pytest.mark.asyncio
async def test_existing_supervisor_agent_still_concatenates_responses(mock_boto3_client):
    lead_agent = make_agent("Supervisor", "Lead response")
    team = [make_agent("Team Member", "Plain response")]
    supervisor = SupervisorAgent(SupervisorAgentOptions(
        name="SupervisorAgent",
        description="My Supervisor agent description",
        lead_agent=lead_agent,
        team=team,
        storage=mock_storage(),
    ))

    response = await supervisor.send_messages([
        {"recipient": "Team Member", "content": "Help"},
    ])

    assert response == "Team Member: Plain response"
    assert "<neutrosophic_consensus>" not in response
