-- Word has no mermaid renderer, so a ```mermaid block prints as literal source.
-- The phase table immediately below the diagram carries the same information, which
-- is why dropping it loses nothing: the diagram is a convenience for readers on
-- GitHub/Obsidian, not the only statement of the chain.
function CodeBlock(el)
  if el.classes:includes('mermaid') then
    return pandoc.Para{
      pandoc.Emph{pandoc.Str('[El diagrama de fases se presenta como tabla a continuación.]')}
    }
  end
  return nil
end
