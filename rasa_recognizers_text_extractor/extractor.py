import logging
import os
import requests
from typing import Any, List, Optional, Text, Dict, Type

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import ENTITIES, TEXT
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.training_data.message import Message
import rasa.shared.utils.io

logger = logging.getLogger(__name__)

def convert_recognizers_format_to_rasa(
    matches: List[Dict[Text, Any]]
) -> List[Dict[Text, Any]]:
    """Converts recognizers to rasa format
    Args:
        matches (List[Dict[Text, Any]]): Matches with entities
    Returns:
        List[Dict[Text, Any]]: List of matches with entities updated to Rasa format
    """
    extracted = []

    for match in matches:
        entity = {
            "start": match.get("start"),
            "end": match.get("end"),
            "text": match.get("text"),
            "value": match.get("resolution", {}).get("value"),
            "confidence": 1.0,
            "additional_info": match.get("resolution"),
            "entity": match.get("typeName"),
        }

        extracted.append(entity)

    return extracted

# Registering component as per Rasa 3.0 requirement
@DefaultV1Recipe.register(
    component_types=[DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR],
    is_trainable=False,
)
class RecognizersTextServiceEntityExtractor(EntityExtractorMixin, GraphComponent):
    """Searches for structured entites, e.g. dates, using recognizers-text-service.
    Args:
        EntityExtractorMixin (rasa.nlu.extractors.extractor): Entity extractor component
        GraphComponent (rasa.engine.graph): Graph interface for component
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Required default configuration method.
        {
            entities: by default all entities recognized by recognizers-text-service are returned.
                entities can be configured to contain an array of strings with the names of
                the entities to filter for
            units: by default all units are returned. Units can be configured to contain an
                array of strings with the names of the units to filter for
            url: http url of the running recognizers-text-service
            culture: if not set, we will use English (en-us)
            show_numbers: a flag to have the service return original numbers in the resolution
            merge_results: a flag to have the service merge overlapping entities
            timeout: Timeout for receiving response from http url of the running
                recognizers-text-service if not set the default timeout of recognizers-text-service url
                is set to 3 seconds.
        }
        Returns:
            Dict[Text, Any]: dict described above
        """
        return {
            "entities": None,
            "units": None,
            "url": None,
            "culture": 'en-us',
            "show_numbers": True,
            "merge_results": True,
            "timeout": 3,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.config = config
        logger.debug(config)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new `RecognizersTextServiceEntityExtractor`.
        Args:
            config (Dict[Text, Any]): Component configuration
        Returns:
            RecognizersTextServiceEntityExtractor: An instantiated `RecognizersTextServiceEntityExtractor`
        """
        return cls(config, execution_context.node_name, model_storage, resource)

    def _url(self) -> Optional[Text]:
        """Return url of recoginzers-service. Environment var will override.
        Returns:
            Optional[Text]: url of recoginzers-service.
        """
        if os.environ.get("RECOGNIZERS_TEXT_SERVICE_URL"):
            return os.environ["RECOGNIZERS_TEXT_SERVICE_URL"]

        return self.config.get("url")

    def _payload(self, text: Text) -> Dict[Text, Any]:
        """Creates payload object and returns it
        Args:
            text (Text): text to include in payload
        Returns:
            Dict[Text, Any]: Payload object
        """
        return {
            "text": text,
            "culture": self.config.get("culture"),
            "entities": self.config.get("entities"),
            "units": self.config.get("units"),
            "showNumbers": self.config.get("show_numbers"),
            "mergeResults": self.config.get("merge_results"),
        }

    def _recognizers_parse(self, text: Text) -> List[Dict[Text, Any]]:
        """Sends the request to recognizers-text-service and parses the result.
        Args:
            text: Text for recognizers-text-service server to parse.
            reference_time: Reference time in milliseconds.
        Returns:
            JSON response from recognizers-text-service with parse data.
        """
        try:
            payload = self._payload(text)
            headers = {
                "Content-Type": "application/json"
            }
            response = requests.post(
                self._url(),
                json=payload,
                headers=headers,
                timeout=self.config.get("timeout"),
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Failed to get a proper response from remote "
                    f"recognizers-text-service at '{self._url()}. Status Code: \
                    {response.status_code}. Response: {response.text}"
                )
                return []
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as e:
            logger.error(
                "Failed to connect to recognizers-text-service. Make sure "
                "the recognizers-text-service is running/healthy/not stale and the proper host "
                "and port are set in the configuration. More "
                "information on how to run the server can be found on "
                "github: "
                "https://github.com/xanthous-tech/recognizers-text-service "
                "Error: {}".format(e)
            )
            return []

    def process(self, messages: List[Message], **kwargs: Any) -> List[Message]:
        """Recognizers extractor method called by Rasa Open Source during inference.
        Args:
            messages (List[Message]): List of Message objects to extract entities from
        """
        for message in messages:
            if self._url() is not None:
                matches = self._recognizers_parse(message.get(TEXT))
                all_extracted = convert_recognizers_format_to_rasa(matches)
                entities = self.config.get("entities")
                extracted = RecognizersTextServiceEntityExtractor.filter_irrelevant_entities(
                    all_extracted, entities
                )
            else:
                extracted = []
                rasa.shared.utils.io.raise_warning(
                    "recognizers-text-service component in pipeline, but no "
                    "`url` configuration in the config "
                    "file nor is `RECOGNIZERS_TEXT_SERVICE_URL` "
                    "set as an environment variable. No entities will be extracted!",
                    docs="https://github.com/botisan-ai/recognizers-text-service",
                )

            extracted = self.add_extractor_name(extracted)
            message.set(ENTITIES, message.get(ENTITIES, []) + extracted, add_to_output=True)

        return messages
