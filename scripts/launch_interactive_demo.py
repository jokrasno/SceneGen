"""Launch interactive_demo.py with compatibility patches for this environment."""

import runpy
import sys
from pathlib import Path

import gradio.blocks
import gradio.networking
import gradio_client.utils as gradio_client_utils
from starlette.templating import Jinja2Templates


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# This pod is reached through Tailscale Serve, so Gradio's localhost reachability
# check can produce a false negative during launch.
gradio.networking.url_ok = lambda *args, **kwargs: True

_original_template_response = Jinja2Templates.TemplateResponse
_original_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type
_original_get_api_info = gradio.blocks.Blocks.get_api_info
_original_get_config_file = gradio.blocks.Blocks.get_config_file


def _compat_template_response(self, *args, **kwargs):
    if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
        name = args[0]
        context = args[1]
        request = context.get("request")
        if request is not None:
            return _original_template_response(self, request, name, context, **kwargs)
    return _original_template_response(self, *args, **kwargs)


def _compat_json_schema_to_python_type(schema, defs):
    if not isinstance(schema, dict):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)


def _state_parameter(index):
    return {
        "label": f"State {index}",
        "parameter_name": f"_state_{index}",
        "parameter_has_default": True,
        "parameter_default": None,
        "type": {},
        "python_type": {"type": "Any", "description": ""},
        "component": "State",
        "example_input": None,
    }


def _state_return(index):
    return {
        "label": f"State {index}",
        "type": {},
        "python_type": {"type": "Any", "description": ""},
        "component": "State",
    }


def _pad_endpoint(endpoint, key, target_length, factory):
    values = endpoint.get(key)
    if not isinstance(values, list):
        return
    while len(values) < target_length:
        values.append(factory(len(values) + 1))


def _sanitize_json_schema(value):
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if key == "additionalProperties" and isinstance(child, bool):
                value[key] = {}
            else:
                _sanitize_json_schema(child)
    elif isinstance(value, list):
        for child in value:
            _sanitize_json_schema(child)


def _compat_get_api_info(self, *args, **kwargs):
    info = _original_get_api_info(self, *args, **kwargs)
    if not isinstance(info, dict):
        return info

    dependencies = self.config.get("dependencies", [])
    for dependency in dependencies:
        api_name = dependency.get("api_name")
        if not api_name:
            continue
        for endpoint_group in ("named_endpoints", "unnamed_endpoints"):
            endpoint = info.get(endpoint_group, {}).get(f"/{api_name}")
            if not endpoint:
                continue
            _pad_endpoint(endpoint, "parameters", len(dependency.get("inputs", [])), _state_parameter)
            _pad_endpoint(endpoint, "returns", len(dependency.get("outputs", [])), _state_return)
    return info


def _compat_get_config_file(self, *args, **kwargs):
    config = _original_get_config_file(self, *args, **kwargs)
    _sanitize_json_schema(config)
    return config


Jinja2Templates.TemplateResponse = _compat_template_response
gradio_client_utils._json_schema_to_python_type = _compat_json_schema_to_python_type
gradio.blocks.Blocks.get_api_info = _compat_get_api_info
gradio.blocks.Blocks.get_config_file = _compat_get_config_file

runpy.run_path(str(REPO_ROOT / "interactive_demo.py"), run_name="__main__")
