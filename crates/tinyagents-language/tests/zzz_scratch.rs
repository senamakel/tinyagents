#[test]
fn scratch_timeout_literal() {
    let src = "graph g { start a node a { kind model timeout 30 next END } }";
    let prog = tinyagents_language::parser::parse_str(src).unwrap();
    let bp = tinyagents_language::compiler::compile(&prog).unwrap();
    println!("NUM timeout: {:?}", bp[0].nodes[0].timeout);

    let src2 = "graph g { start a node a { kind model timeout \"30s\" next END } }";
    let prog2 = tinyagents_language::parser::parse_str(src2).unwrap();
    let bp2 = tinyagents_language::compiler::compile(&prog2).unwrap();
    println!("STR timeout: {:?}", bp2[0].nodes[0].timeout);

    let src3 = "graph g { start a node a { kind model retry { max_attempts: 3, backoff: \"exponential\" } next END } }";
    let prog3 = tinyagents_language::parser::parse_str(src3).unwrap();
    let bp3 = tinyagents_language::compiler::compile(&prog3).unwrap();
    println!("retry: {:?}", bp3[0].nodes[0].retry);

    let src4 = "graph g { start a node a { kind model options [\"yes\", \"no\"] next END } }";
    let prog4 = tinyagents_language::parser::parse_str(src4).unwrap();
    let bp4 = tinyagents_language::compiler::compile(&prog4).unwrap();
    println!("options: {:?}", bp4[0].nodes[0].options);

    let src5 = "graph g { start a node a { kind model next b } node b { kind model sends [send c] next END } node c { kind model next END } }";
    let prog5 = tinyagents_language::parser::parse_str(src5);
    println!("sends parse: {:?}", prog5.is_ok());
}
