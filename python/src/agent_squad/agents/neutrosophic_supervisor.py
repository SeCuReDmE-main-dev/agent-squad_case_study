import asyncio

from agent_squad.agents.agent import Agent
from agent_squad.agents.supervisor_agent import SupervisorAgent, SupervisorAgentOptions
from agent_squad.neutrosophic import Triplet, decide, neutrosophic_evidence_consensus, score_text_response


class NeutrosophicSupervisor(SupervisorAgent):
    """Supervisor agent that adds neutrosophic consensus to team responses."""

    async def send_messages(self, messages: list[dict[str, str]]) -> str:
        """Process messages and return agent responses with neutrosophic consensus."""
        self._validate_messages(messages)
        tasks = [
            asyncio.create_task(
                asyncio.to_thread(
                    self._send_scored_message,
                    agent,
                    message.get('content'),
                )
            )
            for agent in self.team
            for message in messages
            if agent.name == message.get('recipient')
        ]

        if not tasks:
            return self._format_neutrosophic_response(
                [],
                Triplet(T=0.0, I=1.0, F=0.0),
                decide(Triplet(T=0.0, I=1.0, F=0.0)).value,
            )

        scored_responses = await asyncio.gather(*tasks)
        consensus = neutrosophic_evidence_consensus(score for _, _, score in scored_responses)
        action = decide(consensus)

        return self._format_neutrosophic_response(scored_responses, consensus, action.value)

    def _send_scored_message(self, agent: Agent, content: str) -> tuple[str, str, Triplet]:
        raw_response = self.send_message(
            agent=agent,
            content=content,
            user_id=self.user_id,
            session_id=self.session_id,
            additional_params=self.additional_params,
        )
        prefix = f"{agent.name}: "
        response_text = raw_response[len(prefix):] if raw_response.startswith(prefix) else raw_response
        return agent.name, response_text, score_text_response(response_text)

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f"messages[{index}] must be a dictionary")

            recipient = message.get("recipient")
            content = message.get("content")
            if not isinstance(recipient, str) or not recipient.strip():
                raise ValueError(f"messages[{index}].recipient must be a non-empty string")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"messages[{index}].content must be a non-empty string")

    @staticmethod
    def _format_neutrosophic_response(scored_responses, consensus, action: str) -> str:
        score_lines = [
            f"- {agent_name}: {response_text}\n"
            f"  T={score.T:.2f}, I={score.I:.2f}, F={score.F:.2f}"
            for agent_name, response_text, score in scored_responses
        ]

        return (
            "<neutrosophic_consensus>\n"
            f"action: {action}\n"
            f"T: {consensus.T:.2f}\n"
            f"I: {consensus.I:.2f}\n"
            f"F: {consensus.F:.2f}\n"
            "</neutrosophic_consensus>\n"
            "<agent_responses>\n"
            + "\n".join(score_lines)
            + "\n</agent_responses>"
        )
