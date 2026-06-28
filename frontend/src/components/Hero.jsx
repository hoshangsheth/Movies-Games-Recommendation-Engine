import "./Hero.css";

/**
 * Page header block. Used on Home (larger variant) and the
 * Movies/Games/Contact pages (standard variant), matching the per-page
 * hero markup that was duplicated inline across `app.py`'s `elif`
 * branches in the original.
 */
export default function Hero({ eyebrow, title, titleAccent, subtitle, variant = "default" }) {
  const isHome = variant === "home";

  return (
    <header className={`hero${isHome ? " hero-home" : ""}`}>
      {eyebrow && <div className="hero-eyebrow">◈ &nbsp; {eyebrow}</div>}
      <h1 className={`hero-title${isHome ? " hero-title-home" : ""}`}>
        {title}
        {titleAccent && (
          <>
            <br />
            <span className="hero-title-accent">{titleAccent}</span>
          </>
        )}
      </h1>
      {subtitle && <p className={`hero-subtitle${isHome ? " hero-subtitle-home" : ""}`}>{subtitle}</p>}
    </header>
  );
}
