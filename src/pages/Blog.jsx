import { useEffect, useState } from "react";
import { ArrowLeft, ArrowUpRight, Search, X } from "lucide-react";
import posts, { tags as topics } from "virtual:blog-posts";
import s from "./Blog.module.css";

const base = import.meta.env.BASE_URL;
const formatDate = (date) =>
  new Date(`${date}T00:00:00Z`).toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });

function PostMeta({ post }) {
  return (
    <div className={s.meta}>
      {post.date && <time dateTime={post.date}>{formatDate(post.date)}</time>}
      <span>{post.authors.join(", ")}</span>
      <span>{post.readingMinutes} min read</span>
    </div>
  );
}

export default function Blog({ slug }) {
  const [topic, setTopic] = useState("All Posts");
  const [query, setQuery] = useState("");
  const post = slug ? posts.find((item) => item.slug === slug) : null;
  useEffect(() => {
    document.title = `${slug ? post?.title || "Post not found" : "Blog"} | ServingStudio`;
    const description = document.querySelector('meta[name="description"]');
    const previous = description?.content;
    if (description)
      description.content =
        post?.description ||
        "Research, engineering, and updates from ServingStudio.";
    return () => {
      if (description) description.content = previous;
    };
  }, [post, slug]);

  if (slug && !post)
    return (
      <section className={`wrap ${s.notFound}`}>
        <h1>Post not found</h1>
        <a className={s.back} href={`${base}blog.html`}>
          <ArrowLeft size={18} /> Back to the blog
        </a>
      </section>
    );

  if (post)
    return (
      <article className={`wrap ${s.article}`}>
        <a className={s.back} href={`${base}blog.html`}>
          <ArrowLeft size={18} aria-hidden="true" /> All Posts
        </a>
        <header className={s.articleHeader}>
          <div className={s.tags}>
            {post.tags.map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
          <h1>{post.title}</h1>
          <p>{post.description}</p>
          <PostMeta post={post} />
        </header>
        <div className={s.articleLayout}>
          <aside className={s.contents}>
            <nav aria-label="On this page">
              <h2>On this page</h2>
              {post.headings.map(({ id, text }) => (
                <a key={id} href={`#${id}`}>
                  {text}
                </a>
              ))}
            </nav>
          </aside>
          {/* Rendered at build time by react-markdown, without raw HTML support. */}
          <div
            className={s.prose}
            dangerouslySetInnerHTML={{ __html: post.html }}
          />
        </div>
        <a className={s.back} href={`${base}blog.html`}>
          <ArrowLeft size={18} aria-hidden="true" /> Back to all posts
        </a>
      </article>
    );

  const normalizedQuery = query.trim().toLocaleLowerCase();
  const filtered = posts.filter(
    (item) =>
      (topic === "All Posts" || item.tags.includes(topic)) &&
      [item.title, item.description, ...item.tags, ...item.authors]
        .join(" ")
        .toLocaleLowerCase()
        .includes(normalizedQuery),
  );
  const reset = () => {
    setTopic("All Posts");
    setQuery("");
  };

  return (
    <>
      <header className={s.hero}>
        <img
          className={s.heroImage}
          src={`${base}images/hero-blog.webp`}
          srcSet={`${base}images/hero-blog-960.webp 960w, ${base}images/hero-blog.webp 1672w`}
          sizes="100vw"
          width="1672"
          height="955"
          alt=""
          fetchPriority="high"
        />
        <div className={`wrap ${s.heroContent}`}>
          <h1>Blog</h1>
        </div>
      </header>
      <section className={`wrap ${s.archive}`} aria-label="Blog posts">
        <div className={s.toolbar}>
          <div className={s.filters} role="group" aria-label="Filter by topic">
            {["All Posts", ...topics].map((name) => (
              <button
                key={name}
                type="button"
                aria-pressed={topic === name}
                onClick={() => setTopic(name)}
              >
                {name}
              </button>
            ))}
          </div>
          <div className={s.search} role="search">
            <Search size={18} aria-hidden="true" />
            <input
              type="search"
              aria-label="Search posts"
              placeholder="Search posts"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            {query && (
              <button
                type="button"
                aria-label="Clear search"
                onClick={() => setQuery("")}
              >
                <X size={16} />
              </button>
            )}
          </div>
        </div>
        <div className={s.resultHeader}>
          <h2>{topic === "All Posts" ? "Latest posts" : topic}</h2>
          <span role="status">
            {filtered.length} {filtered.length === 1 ? "post" : "posts"}
          </span>
        </div>
        {filtered.length ? (
          <ul className={s.posts}>
            {filtered.map((item) => (
              <li key={item.slug}>
                <article>
                  <a href={item.href} className={s.postLink}>
                    <div className={s.cover}>
                      <img
                        src={item.cover}
                        alt={item.coverAlt}
                        width="1672"
                        height="941"
                        loading="lazy"
                      />
                    </div>
                    <div className={s.postCopy}>
                      <div className={s.tags}>
                        {item.tags.map((tag) => (
                          <span key={tag}>{tag}</span>
                        ))}
                      </div>
                      <h3>{item.title}</h3>
                      <p>{item.description}</p>
                      <PostMeta post={item} />
                    </div>
                  </a>
                </article>
              </li>
            ))}
          </ul>
        ) : (
          <div className={s.empty}>
            <Search size={28} aria-hidden="true" />
            <h3>No posts found</h3>
            <p>Try another search or choose a different topic.</p>
            <button type="button" onClick={reset}>
              Clear filters <ArrowUpRight size={18} aria-hidden="true" />
            </button>
          </div>
        )}
      </section>
    </>
  );
}
