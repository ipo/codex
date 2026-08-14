use crate::function_tool::FunctionCallError;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::ToolSearchOutput;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::tool_search_spec::ToolSearchSourceListing;
use crate::tools::handlers::tool_search_spec::create_native_tool_search_tool;
use crate::tools::handlers::tool_search_spec::create_tool_search_tool;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;
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
use codex_tools::serialize_loadable_tool_specs;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::instrument;

pub struct ToolSearchHandler {
    index: Arc<ToolSearchIndex>,
    namespace_tool_spec_mode: NamespaceToolSpecMode,
    native_loaded_tools: Option<Arc<NativeToolSearchState>>,
    spec: ToolSpec,
}

pub(crate) struct ToolSearchIndex {
    search_infos: Vec<ToolSearchInfo>,
    search_engine: SearchEngine<usize>,
}

#[derive(Default)]
pub(crate) struct NativeToolSearchState {
    loaded_tools: Mutex<Vec<ToolSpec>>,
}

impl NativeToolSearchState {
    pub(crate) fn loaded_tools(&self) -> Vec<ToolSpec> {
        self.loaded_tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn replace(&self, tools: Vec<ToolSpec>) {
        *self
            .loaded_tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = tools;
    }
}

#[derive(Default)]
pub(crate) struct ToolSearchHandlerCache {
    cached: Mutex<Option<Arc<ToolSearchIndex>>>,
    native_loaded_tools: Mutex<Option<(String, Arc<NativeToolSearchState>)>>,
}

impl ToolSearchHandlerCache {
    #[instrument(level = "trace", skip_all, fields(search_info_count = search_infos.len()))]
    pub(crate) fn get_or_build(&self, search_infos: Vec<ToolSearchInfo>) -> Arc<ToolSearchIndex> {
        {
            let cached = self.cached();
            if let Some(cached) = cached.as_ref()
                && cached.search_infos == search_infos
            {
                return Arc::clone(cached);
            }
        }

        let index = Arc::new(ToolSearchIndex::new(search_infos));
        let mut cached = self.cached();
        if let Some(cached) = cached.as_ref()
            && cached.search_infos == index.search_infos
        {
            return Arc::clone(cached);
        }

        *cached = Some(Arc::clone(&index));
        index
    }

    fn cached(&self) -> std::sync::MutexGuard<'_, Option<Arc<ToolSearchIndex>>> {
        match self.cached.lock() {
            Ok(cached) => cached,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    pub(crate) fn native_loaded_tools_for_turn(&self, turn_id: &str) -> Arc<NativeToolSearchState> {
        let mut cached = self
            .native_loaded_tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((cached_turn_id, state)) = cached.as_ref()
            && cached_turn_id == turn_id
        {
            return Arc::clone(state);
        }

        let state = Arc::new(NativeToolSearchState::default());
        *cached = Some((turn_id.to_string(), Arc::clone(&state)));
        state
    }
}

impl ToolSearchHandler {
    #[instrument(
        level = "trace",
        skip_all,
        fields(search_info_count = index.search_infos.len())
    )]
    pub(crate) fn new(
        index: Arc<ToolSearchIndex>,
        source_listing: ToolSearchSourceListing,
        namespace_tool_spec_mode: NamespaceToolSpecMode,
        native_loaded_tools: Option<Arc<NativeToolSearchState>>,
    ) -> Self {
        let search_source_infos = index
            .search_infos
            .iter()
            .filter_map(|search_info| search_info.source_info.clone())
            .collect::<Vec<_>>();
        let spec = if native_loaded_tools.is_some() {
            create_native_tool_search_tool(
                &search_source_infos,
                TOOL_SEARCH_DEFAULT_LIMIT,
                source_listing,
            )
        } else {
            create_tool_search_tool(
                &search_source_infos,
                TOOL_SEARCH_DEFAULT_LIMIT,
                source_listing,
            )
        };

        Self {
            index,
            namespace_tool_spec_mode,
            native_loaded_tools,
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

    fn handle(&self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'_> {
        Box::pin(self.handle_call(invocation))
    }
}

impl ToolSearchHandler {
    async fn handle_call(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ToolInvocation { payload, turn, .. } = invocation;

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
        let limit = args.limit.unwrap_or(TOOL_SEARCH_DEFAULT_LIMIT);

        if limit == 0 {
            return Err(FunctionCallError::RespondToModel(
                "limit must be greater than zero".to_string(),
            ));
        }

        let limit = if self.native_loaded_tools.is_some() {
            limit.min(TOOL_SEARCH_DEFAULT_LIMIT)
        } else {
            limit
        };
        let tools = if self.index.search_infos.is_empty() {
            Vec::new()
        } else {
            self.search(query, limit)?
        };

        if let Some(native_loaded_tools) = &self.native_loaded_tools {
            let loaded_tools = tools
                .into_iter()
                .map(|tool| match tool {
                    LoadableToolSpec::Function(mut tool) => {
                        tool.defer_loading = None;
                        ToolSpec::Function(tool)
                    }
                    LoadableToolSpec::Namespace(_) => {
                        unreachable!("native tool search must flatten namespace tools")
                    }
                })
                .map(|spec| crate::tools::wire_adaptation::adapt_spec_for_wire(turn.as_ref(), spec))
                .collect::<Vec<_>>();
            let loaded_tools =
                crate::tools::wire_adaptation::validate_model_visible_function_names(loaded_tools);
            let output = serde_json::json!({"tools": loaded_tools}).to_string();
            native_loaded_tools.replace(loaded_tools);
            return Ok(boxed_tool_output(
                crate::tools::context::FunctionToolOutput::from_text(output, Some(true)),
            ));
        }

        Ok(boxed_tool_output(ToolSearchOutput { tools }))
    }
}

impl CoreToolRuntime for ToolSearchHandler {}

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
        Ok(serialize_loadable_tool_specs(
            results.into_iter().map(|entry| entry.output.clone()),
            self.namespace_tool_spec_mode,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::step_context::StepContext;
    use crate::session::tests::make_session_and_context;
    use crate::tools::context::ToolCallSource;
    use crate::tools::handlers::DynamicToolHandler;
    use crate::tools::handlers::McpHandler;
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
    fn cache_reuses_index_for_identical_search_infos_and_rebuilds_for_changes() {
        let cache = ToolSearchHandlerCache::default();
        let search_infos = vec![
            McpHandler::new(tool_info("calendar", "create_event", "Create events"))
                .expect("MCP tool should convert")
                .search_info()
                .expect("MCP handler should return search info"),
        ];

        let first = cache.get_or_build(search_infos.clone());
        let second = cache.get_or_build(search_infos.clone());
        assert!(Arc::ptr_eq(&first, &second));

        let mut changed_search_infos = search_infos;
        changed_search_infos[0]
            .entry
            .search_text
            .push_str(" changed");
        let changed = cache.get_or_build(changed_search_infos);
        assert!(!Arc::ptr_eq(&first, &changed));
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
        let handler = ToolSearchHandler::new(
            Arc::clone(&index),
            ToolSearchSourceListing::Include,
            NamespaceToolSpecMode::Preserve,
            None,
        );
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
                            local_result_schema: Some(
                                codex_tools::mcp_call_tool_result_output_schema(serde_json::json!(
                                    {}
                                ),),
                            ),
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
                            local_result_schema: Some(
                                codex_tools::mcp_call_tool_result_output_schema(serde_json::json!(
                                    {}
                                ),),
                            ),
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
                        local_result_schema: None,
                    })],
                }),
            ],
        );
    }

    #[test]
    fn native_search_flattens_namespaces_and_replaces_loaded_state() {
        let search_info = McpHandler::new(tool_info("calendar", "create_event", "Create events"))
            .expect("MCP tool should convert")
            .search_info()
            .expect("MCP handler should return search info");
        let index = Arc::new(ToolSearchIndex::new(vec![search_info]));
        let cache = ToolSearchHandlerCache::default();
        let state = cache.native_loaded_tools_for_turn("turn-1");
        let handler = ToolSearchHandler::new(
            Arc::clone(&index),
            ToolSearchSourceListing::Include,
            NamespaceToolSpecMode::Flatten,
            Some(Arc::clone(&state)),
        );
        let mut tools = handler
            .search_output_tools([&index.search_infos[0].entry])
            .expect("MCP search output should serialize")
            .into_iter()
            .map(|tool| match tool {
                LoadableToolSpec::Function(mut tool) => {
                    tool.defer_loading = None;
                    ToolSpec::Function(tool)
                }
                LoadableToolSpec::Namespace(_) => panic!("native result must be flat"),
            })
            .collect::<Vec<_>>();
        state.replace(tools.clone());

        assert_eq!(state.loaded_tools(), tools);
        let ToolSpec::Function(tool) = tools.remove(0) else {
            panic!("loaded tool must be a function");
        };
        assert_eq!(tool.name, "mcp__calendar__create_event");
        assert_eq!(tool.defer_loading, None);

        let rebuilt_state = cache.native_loaded_tools_for_turn("turn-1");
        assert!(Arc::ptr_eq(&state, &rebuilt_state));
        assert_eq!(rebuilt_state.loaded_tools().len(), 1);

        let next_turn_state = cache.native_loaded_tools_for_turn("turn-2");
        let _next_turn_handler = ToolSearchHandler::new(
            index,
            ToolSearchSourceListing::Include,
            NamespaceToolSpecMode::Flatten,
            Some(Arc::clone(&next_turn_state)),
        );
        assert_eq!(next_turn_state.loaded_tools(), Vec::<ToolSpec>::new());
    }

    #[tokio::test]
    async fn native_search_caps_and_replaces_loaded_tools() {
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
        let state = Arc::new(NativeToolSearchState::default());
        let handler = ToolSearchHandler::new(
            Arc::new(ToolSearchIndex::new(search_infos)),
            ToolSearchSourceListing::Include,
            NamespaceToolSpecMode::Flatten,
            Some(Arc::clone(&state)),
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
            source: ToolCallSource::Direct,
            payload: ToolPayload::Function {
                arguments: serde_json::json!({
                    "query": "shared searchable calendar action",
                    "limit": 20,
                })
                .to_string(),
            },
        };

        handler
            .handle_call(invocation.clone())
            .await
            .expect("native search should succeed");
        assert_eq!(state.loaded_tools().len(), TOOL_SEARCH_DEFAULT_LIMIT);

        let replacement = ToolSearchHandler::new(
            Arc::new(ToolSearchIndex::new(vec![
                McpHandler::new(tool_info("mail", "send", "send mail"))
                    .expect("MCP tool should convert")
                    .search_info()
                    .expect("MCP handler should return search info"),
            ])),
            ToolSearchSourceListing::Include,
            NamespaceToolSpecMode::Flatten,
            Some(Arc::clone(&state)),
        );
        replacement
            .handle_call(ToolInvocation {
                payload: ToolPayload::Function {
                    arguments: serde_json::json!({"query": "send mail"}).to_string(),
                },
                ..invocation
            })
            .await
            .expect("replacement search should succeed");

        assert_eq!(state.loaded_tools().len(), 1);
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
