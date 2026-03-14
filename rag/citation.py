def format_sources(chunks, max_sources=3):

    sources = []
    seen = set()

    for chunk in chunks:

        source = f"{chunk['source']} (page {chunk['page']})"

        if source not in seen:
            sources.append(source)
            seen.add(source)

        if len(sources) >= max_sources:
            break

    return sources