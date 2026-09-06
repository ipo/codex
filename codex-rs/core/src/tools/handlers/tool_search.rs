use crate::function_tool::FunctionCallError;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::ToolSearchOutput;
use crate::tools::context::boxed_tool_output;
use crate::tools::deferred_tool_state::DeferredToolLoadState;
use crate::tools::handlers::tool_search_spec::ToolSearchSourceListing;
use crate::tools::handlers::tool_search_spec::create_function_tool_search_tool;
use crate::tools::handlers::tool_search_spec::create_tool_search_tool;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;
use crate::tools::registry::ToolRegistry;
use bm25::Document;
use bm25::Language;
use bm25::SearchEngine;
use bm25::SearchEngineBuilder;
use codex_tools::LoadableToolSpec;
use codex_tools::NamespaceToolSpecMode;
use codex_tools::TOOL_SEARCH_DEFAULT_LIMIT;
use codex_tools::TOOL_SEARCH_TOOL_NAME;
use codex_tools::ToolName;
use codex_tools::ToolSearchEntry;
use codex_tools::ToolSearchInfo;
use codex_tools::ToolSpec;
use codex_tools::coalesce_loadable_tool_specs;
use codex_tools::serialize_tool_specs;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;
use tracing::instrument;

pub struct ToolSearchHandler {
    index: Arc<ToolSearchIndex>,
    spec_mode: ToolSearchSpecMode,
    spec: ToolSpec,
}

pub(crate) struct ToolSearchIndex {
    search_infos: Vec<ToolSearchInfo>,
    search_engine: SearchEngine<usize>,
}

enum ToolSearchSpecMode {
    Responses,
    Function(Arc<DeferredToolLoadState>),
}

#[derive(Default)]
pub(crate) struct ToolSearchHandlerCache {
    cached: Mutex<Option<CachedToolSearchIndex>>,
}

struct CachedToolSearchIndex {
    index: Arc<ToolSearchIndex>,
    sources: Vec<ToolSearchSource>,
}

enum ToolSearchSource {
    Immutable(Weak<dyn CoreToolRuntime>),
    Dynamic(Box<ToolSearchInfo>),
}

impl ToolSearchHandlerCache {
    #[instrument(level = "trace", skip_all)]
    pub(crate) fn get_or_build(&self, registry: &ToolRegistry) -> Arc<ToolSearchIndex> {
        let sources = registry
            .entries()
            .filter(|tool| tool.exposure.is_deferred())
            .filter_map(|tool| {
                if tool.runtime.immutable_spec().is_some() {
                    Some(ToolSearchSource::Immutable(Arc::downgrade(&tool.runtime)))
                } else {
                    tool.runtime
                        .search_info()
                        .map(Box::new)
                        .map(ToolSearchSource::Dynamic)
                }
            })
            .collect::<Vec<_>>();

        {
            let cached = self.cached();
            if let Some(cached) = cached.as_ref()
                && Self::sources_match(&cached.sources, &sources)
            {
                return Arc::clone(&cached.index);
            }
        }

        let search_infos = sources
            .iter()
            .filter_map(|source| match source {
                ToolSearchSource::Immutable(runtime) => {
                    runtime.upgrade().and_then(|runtime| runtime.search_info())
                }
                ToolSearchSource::Dynamic(search_info) => Some(search_info.as_ref().clone()),
            })
            .collect();

        let index = Arc::new(ToolSearchIndex::new(search_infos));
        let mut cached = self.cached();
        if let Some(cached) = cached.as_ref()
            && Self::sources_match(&cached.sources, &sources)
        {
            return Arc::clone(&cached.index);
        }
        *cached = Some(CachedToolSearchIndex {
            index: Arc::clone(&index),
            sources,
        });
        index
    }

    fn sources_match(cached_sources: &[ToolSearchSource], sources: &[ToolSearchSource]) -> bool {
        cached_sources.len() == sources.len()
            && cached_sources
                .iter()
                .zip(sources)
                .all(|(cached, current)| match (cached, current) {
                    (ToolSearchSource::Immutable(cached), ToolSearchSource::Immutable(current)) => {
                        Weak::ptr_eq(cached, current)
                    }
                    (ToolSearchSource::Dynamic(cached), ToolSearchSource::Dynamic(current)) => {
                        cached == current
                    }
                    (ToolSearchSource::Immutable(_), ToolSearchSource::Dynamic(_))
                    | (ToolSearchSource::Dynamic(_), ToolSearchSource::Immutable(_)) => false,
                })
    }

    fn cached(&self) -> std::sync::MutexGuard<'_, Option<CachedToolSearchIndex>> {
        match self.cached.lock() {
            Ok(cached) => cached,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl ToolSearchHandler {
    #[instrument(
        level = "trace",
        skip_all,
        fields(search_info_count = index.search_infos.len())
    )]
    pub(crate) fn responses(
        index: Arc<ToolSearchIndex>,
        source_listing: ToolSearchSourceListing,
    ) -> Self {
        let search_source_infos = index
            .search_infos
            .iter()
            .filter_map(|search_info| search_info.source_info.clone())
            .collect::<Vec<_>>();
        let spec = create_tool_search_tool(
            &search_source_infos,
            TOOL_SEARCH_DEFAULT_LIMIT,
            source_listing,
        );
        Self {
            index,
            spec_mode: ToolSearchSpecMode::Responses,
            spec,
        }
    }

    pub(crate) fn function(
        index: Arc<ToolSearchIndex>,
        source_listing: ToolSearchSourceListing,
        state: Arc<DeferredToolLoadState>,
    ) -> Self {
        let search_source_infos = index
            .search_infos
            .iter()
            .filter_map(|search_info| search_info.source_info.clone())
            .collect::<Vec<_>>();
        let spec = create_function_tool_search_tool(
            &search_source_infos,
            TOOL_SEARCH_DEFAULT_LIMIT,
            source_listing,
        );
        Self {
            index,
            spec_mode: ToolSearchSpecMode::Function(state),
            spec,
        }
    }
}

impl ToolSearchIndex {
    fn new(search_infos: Vec<ToolSearchInfo>) -> Self {
        let documents: Vec<Document<usize>> = search_infos
            .iter()
            .map(|search_info| search_info.entry.search_text.clone())
            .enumerate()
            .map(|(idx, search_text)| Document::new(idx, search_text))
            .collect();
        let search_engine =
            SearchEngineBuilder::<usize>::with_documents(Language::English, documents).build();

        Self {
            search_infos,
            search_engine,
        }
    }
}

impl ToolExecutor<ToolInvocation> for ToolSearchHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(TOOL_SEARCH_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }

    fn handle<'a>(&'a self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'a>
    where
        ToolInvocation: 'a,
    {
        Box::pin(self.handle_call(invocation))
    }
}

impl ToolSearchHandler {
    async fn handle_call(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ToolInvocation {
            payload,
            step_context,
            ..
        } = invocation;

        let args = match payload {
            ToolPayload::ToolSearch { arguments } => arguments,
            ToolPayload::Function { arguments } => {
                serde_json::from_str(&arguments).map_err(|err| {
                    FunctionCallError::RespondToModel(format!(
                        "failed to parse {TOOL_SEARCH_TOOL_NAME} arguments: {err}"
                    ))
                })?
            }
            ToolPayload::Custom { .. } => {
                return Err(FunctionCallError::Fatal(format!(
                    "{TOOL_SEARCH_TOOL_NAME} handler received unsupported payload"
                )));
            }
        };

        let query = args.query.trim();
        if query.is_empty() {
            return Err(FunctionCallError::RespondToModel(
                "query must not be empty".to_string(),
            ));
        }
        let mut limit = args.limit.unwrap_or(TOOL_SEARCH_DEFAULT_LIMIT);

        if limit == 0 {
            return Err(FunctionCallError::RespondToModel(
                "limit must be greater than zero".to_string(),
            ));
        }

        if matches!(&self.spec_mode, ToolSearchSpecMode::Function(_)) {
            limit = limit.min(TOOL_SEARCH_DEFAULT_LIMIT);
        }

        let tools = if self.index.search_infos.is_empty() {
            Vec::new()
        } else {
            self.search(query, limit)?
        };

        if let ToolSearchSpecMode::Function(state) = &self.spec_mode {
            let loaded_tools = serialize_tool_specs(
                tools.into_iter().map(ToolSpec::from),
                NamespaceToolSpecMode::Flatten,
            )
            .into_iter()
            .map(|mut spec| {
                match &mut spec {
                    ToolSpec::Function(tool) => tool.defer_loading = None,
                    ToolSpec::Freeform(tool) => tool.defer_loading = None,
                    ToolSpec::Namespace(_)
                    | ToolSpec::ToolSearch { .. }
                    | ToolSpec::WebSearch { .. } => {
                        unreachable!("flattened deferred search results must be callable tools")
                    }
                }
                crate::tools::wire_adaptation::adapt_spec_for_wire(
                    &step_context.turn,
                    &step_context.settings.model_info,
                    spec,
                )
            })
            .collect();
            let loaded_tools =
                crate::tools::wire_adaptation::validate_model_visible_function_names(loaded_tools);
            let output = serde_json::json!({"tools": loaded_tools}).to_string();
            state.replace(loaded_tools);
            return Ok(boxed_tool_output(FunctionToolOutput::from_text(
                output,
                Some(true),
            )));
        }

        Ok(boxed_tool_output(ToolSearchOutput { tools }))
    }
}

impl CoreToolRuntime for ToolSearchHandler {
    fn matches_kind(&self, payload: &ToolPayload) -> bool {
        matches!(
            (&self.spec_mode, payload),
            (
                ToolSearchSpecMode::Responses,
                ToolPayload::ToolSearch { .. }
            ) | (
                ToolSearchSpecMode::Function(_),
                ToolPayload::Function { .. }
            )
        )
    }
}

impl ToolSearchHandler {
    fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<LoadableToolSpec>, FunctionCallError> {
        let results = self
            .index
            .search_engine
            .search(query, limit)
            .into_iter()
            .map(|result| result.document.id)
            .filter_map(|id| self.index.search_infos.get(id))
            .map(|search_info| &search_info.entry);
        self.search_output_tools(results)
    }

    fn search_output_tools<'a>(
        &self,
        results: impl IntoIterator<Item = &'a ToolSearchEntry>,
    ) -> Result<Vec<LoadableToolSpec>, FunctionCallError> {
        Ok(coalesce_loadable_tool_specs(
            results.into_iter().map(|entry| entry.output.clone()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::step_context::StepContext;
    use crate::session::tests::make_session_and_context;
    use crate::tools::handlers::DynamicToolHandler;
    use crate::tools::handlers::McpHandler;
    use crate::tools::registry::ToolExposure;
    use crate::turn_diff_tracker::TurnDiffTracker;
    use codex_mcp::ToolInfo;
    use codex_protocol::dynamic_tools::DynamicToolFunctionSpec;
    use codex_protocol::dynamic_tools::DynamicToolNamespaceSpec;
    use codex_tools::ResponsesApiNamespace;
    use codex_tools::ResponsesApiNamespaceTool;
    use codex_tools::ResponsesApiTool;
    use pretty_assertions::assert_eq;
    use rmcp::model::Tool;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn cache_reuses_immutable_indexes_and_rebuilds_for_current_registry_changes() {
        let cache = ToolSearchHandlerCache::default();
        let runtime: Arc<dyn CoreToolRuntime> = Arc::new(
            McpHandler::new(tool_info("calendar", "create_event", "Create events"))
                .expect("MCP tool should convert"),
        );
        let mut registry = ToolRegistry::default();
        registry.register_trusted_with_exposure(Arc::clone(&runtime), ToolExposure::Deferred);

        let first = cache.get_or_build(&registry);
        let second = cache.get_or_build(&registry);
        assert!(Arc::ptr_eq(&first, &second));

        let mut replacement_registry = ToolRegistry::default();
        let replacement = Arc::new(
            McpHandler::new(tool_info("calendar", "create_event", "Create events"))
                .expect("replacement MCP tool should convert"),
        );
        replacement_registry.register_trusted_with_exposure(replacement, ToolExposure::Deferred);
        let replacement = cache.get_or_build(&replacement_registry);
        assert!(!Arc::ptr_eq(&first, &replacement));

        let mut disabled_registry = ToolRegistry::default();
        disabled_registry.register_trusted_with_exposure(runtime, ToolExposure::Direct);
        let disabled = cache.get_or_build(&disabled_registry);
        assert!(!Arc::ptr_eq(&replacement, &disabled));
        assert!(disabled.search_infos.is_empty());
    }

    #[test]
    fn cache_rechecks_dynamic_tool_metadata_while_reusing_immutable_mcp_handlers() {
        let cache = ToolSearchHandlerCache::default();
        let mcp_runtime: Arc<dyn CoreToolRuntime> = Arc::new(
            McpHandler::new(tool_info("calendar", "create_event", "Create events"))
                .expect("MCP tool should convert"),
        );
        let mut dynamic_tool = DynamicToolFunctionSpec {
            name: "lookup".to_string(),
            description: "Search current records".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
            defer_loading: true,
        };

        let mut first_registry = ToolRegistry::default();
        first_registry
            .register_trusted_with_exposure(Arc::clone(&mcp_runtime), ToolExposure::Deferred);
        first_registry.register_external_with_exposure(
            Arc::new(DynamicToolHandler::new(&dynamic_tool).expect("dynamic tool should convert")),
            ToolExposure::Deferred,
        );
        let first = cache.get_or_build(&first_registry);

        let mut equivalent_registry = ToolRegistry::default();
        equivalent_registry
            .register_trusted_with_exposure(Arc::clone(&mcp_runtime), ToolExposure::Deferred);
        equivalent_registry.register_external_with_exposure(
            Arc::new(DynamicToolHandler::new(&dynamic_tool).expect("dynamic tool should convert")),
            ToolExposure::Deferred,
        );
        let equivalent = cache.get_or_build(&equivalent_registry);
        assert!(Arc::ptr_eq(&first, &equivalent));

        dynamic_tool.description = "Search refreshed records".to_string();
        let mut refreshed_registry = ToolRegistry::default();
        refreshed_registry.register_trusted_with_exposure(mcp_runtime, ToolExposure::Deferred);
        refreshed_registry.register_external_with_exposure(
            Arc::new(DynamicToolHandler::new(&dynamic_tool).expect("dynamic tool should convert")),
            ToolExposure::Deferred,
        );
        let refreshed = cache.get_or_build(&refreshed_registry);
        assert!(!Arc::ptr_eq(&first, &refreshed));
        assert!(
            refreshed.search_infos[1]
                .entry
                .search_text
                .contains("refreshed")
        );
    }

    #[test]
    fn mixed_search_results_coalesce_mcp_namespaces() {
        let dynamic_namespace = DynamicToolNamespaceSpec {
            name: "codex_app".to_string(),
            description: "Tools in the codex_app namespace.".to_string(),
            tools: Vec::new(),
        };
        let dynamic_tools = [DynamicToolFunctionSpec {
            name: "automation_update".to_string(),
            description: "Create, update, view, or delete recurring automations.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "mode": { "type": "string" },
                },
                "required": ["mode"],
                "additionalProperties": false,
            }),
            defer_loading: true,
        }];
        let mcp_tools = [
            tool_info("calendar", "create_event", "Create events"),
            tool_info("calendar", "list_events", "List events"),
        ];
        let mut search_infos = mcp_tools
            .iter()
            .map(|tool| {
                McpHandler::new(tool.clone())
                    .expect("MCP tool should convert")
                    .search_info()
                    .expect("MCP handler should return search info")
            })
            .collect::<Vec<_>>();
        search_infos.extend(dynamic_tools.iter().map(|tool| {
            DynamicToolHandler::new_in_namespace(&dynamic_namespace, tool)
                .expect("dynamic tool should convert")
                .search_info()
                .expect("dynamic handler should return search info")
        }));
        let index = Arc::new(ToolSearchIndex::new(search_infos));
        let handler =
            ToolSearchHandler::responses(Arc::clone(&index), ToolSearchSourceListing::Include);
        let results = [
            &index.search_infos[0].entry,
            &index.search_infos[2].entry,
            &index.search_infos[1].entry,
        ];

        let tools = handler
            .search_output_tools(results)
            .expect("mixed search output should serialize");

        assert_eq!(
            tools,
            vec![
                LoadableToolSpec::Namespace(ResponsesApiNamespace {
                    name: "mcp__calendar".to_string(),
                    description: "Tools in the mcp__calendar namespace.".to_string(),
                    tools: vec![
                        ResponsesApiNamespaceTool::Function(ResponsesApiTool {
                            name: "create_event".to_string(),
                            description: "Create events desktop tool".to_string(),
                            strict: false,
                            defer_loading: Some(true),
                            parameters: codex_tools::JsonSchema::object(
                                Default::default(),
                                /*required*/ None,
                                Some(false.into()),
                            ),
                            output_schema: None,
                        }),
                        ResponsesApiNamespaceTool::Function(ResponsesApiTool {
                            name: "list_events".to_string(),
                            description: "List events desktop tool".to_string(),
                            strict: false,
                            defer_loading: Some(true),
                            parameters: codex_tools::JsonSchema::object(
                                Default::default(),
                                /*required*/ None,
                                Some(false.into()),
                            ),
                            output_schema: None,
                        }),
                    ],
                }),
                LoadableToolSpec::Namespace(ResponsesApiNamespace {
                    name: "codex_app".to_string(),
                    description: "Tools in the codex_app namespace.".to_string(),
                    tools: vec![ResponsesApiNamespaceTool::Function(ResponsesApiTool {
                        name: "automation_update".to_string(),
                        description: "Create, update, view, or delete recurring automations."
                            .to_string(),
                        strict: false,
                        defer_loading: Some(true),
                        parameters: codex_tools::JsonSchema::object(
                            std::collections::BTreeMap::from([(
                                "mode".to_string(),
                                codex_tools::JsonSchema::string(/*description*/ None),
                            )]),
                            Some(vec!["mode".to_string()]),
                            Some(false.into()),
                        ),
                        output_schema: None,
                    })],
                }),
            ],
        );
    }

    #[tokio::test]
    async fn function_search_caps_and_replaces_flattened_loaded_tools() {
        let search_infos = (0..10)
            .map(|index| {
                McpHandler::new(tool_info(
                    "calendar",
                    &format!("tool_{index}"),
                    &format!("shared searchable calendar action {index}"),
                ))
                .expect("MCP tool should convert")
                .search_info()
                .expect("MCP handler should return search info")
            })
            .collect();
        let state = Arc::new(DeferredToolLoadState::default());
        let handler = ToolSearchHandler::function(
            Arc::new(ToolSearchIndex::new(search_infos)),
            ToolSearchSourceListing::Include,
            Arc::clone(&state),
        );
        let (session, turn) = make_session_and_context().await;
        let turn = Arc::new(turn);
        let invocation = ToolInvocation {
            session: Arc::new(session),
            step_context: StepContext::for_test(Arc::clone(&turn)),
            turn,
            cancellation_token: CancellationToken::new(),
            tracker: Arc::new(Mutex::new(TurnDiffTracker::new())),
            call_id: "search-call".to_string(),
            tool_name: ToolName::plain(TOOL_SEARCH_TOOL_NAME),
            source: crate::tools::context::ToolCallSource::Direct,
            payload: ToolPayload::Function {
                arguments: serde_json::json!({
                    "query": "shared searchable calendar action",
                    "limit": 20,
                })
                .to_string(),
            },
        };

        let output = handler
            .handle_call(invocation.clone())
            .await
            .expect("function search should succeed");

        let loaded = state.loaded_tools();
        assert_eq!(loaded.len(), TOOL_SEARCH_DEFAULT_LIMIT);
        assert!(loaded.iter().all(|spec| matches!(
            spec,
            ToolSpec::Function(tool)
                if tool.name.starts_with("mcp__calendar__tool_")
                    && tool.defer_loading.is_none()
        )));
        let codex_protocol::models::ResponseInputItem::FunctionCallOutput { output, .. } =
            output.to_response_item(&invocation.call_id, &invocation.payload)
        else {
            panic!("function search must return an ordinary function output");
        };
        let codex_protocol::models::FunctionCallOutputBody::Text(output) = output.body else {
            panic!("function search output must be plaintext JSON");
        };
        let output: serde_json::Value =
            serde_json::from_str(&output).expect("function search output should be JSON");
        assert_eq!(
            output["tools"]
                .as_array()
                .expect("tool list should be an array")
                .len(),
            TOOL_SEARCH_DEFAULT_LIMIT
        );

        let replacement = ToolSearchHandler::function(
            Arc::new(ToolSearchIndex::new(vec![
                McpHandler::new(tool_info("mail", "send", "send mail"))
                    .expect("MCP tool should convert")
                    .search_info()
                    .expect("MCP handler should return search info"),
            ])),
            ToolSearchSourceListing::Include,
            Arc::clone(&state),
        );
        replacement
            .handle_call(ToolInvocation {
                call_id: "replacement-search-call".to_string(),
                payload: ToolPayload::Function {
                    arguments: serde_json::json!({"query": "send mail"}).to_string(),
                },
                ..invocation
            })
            .await
            .expect("replacement search should succeed");

        let loaded = state.loaded_tools();
        assert_eq!(loaded.len(), 1);
        assert!(matches!(
            &loaded[0],
            ToolSpec::Function(tool) if tool.name == "mcp__mail__send"
        ));
    }

    fn tool_info(server_name: &str, tool_name: &str, description_prefix: &str) -> ToolInfo {
        ToolInfo {
            server_name: server_name.to_string(),
            supports_parallel_tool_calls: false,
            server_origin: None,
            callable_name: tool_name.to_string(),
            callable_namespace: format!("mcp__{server_name}"),
            namespace_description: None,
            tool: Tool::new(
                tool_name.to_string(),
                format!("{description_prefix} desktop tool"),
                Arc::new(rmcp::model::object(serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false,
                }))),
            ),
            openai_file_input_optional_fields: Default::default(),
            connector_id: None,
            connector_name: None,
            plugin_display_names: Vec::new(),
        }
    }
}
