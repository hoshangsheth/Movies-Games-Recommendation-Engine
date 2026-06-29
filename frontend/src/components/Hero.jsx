import { Sparkles } from "lucide-react";
import "./Hero.css";

export default function Hero({ eyebrow, title, titleAccent, subtitle, variant = "default" }) {
  const isHome = variant === "home";

  return (
    <header className={`hero${isHome ? " hero-home" : ""}`}>
      {eyebrow && (
        <div className="hero-eyebrow">
          <Sparkles size={13} strokeWidth={2.5} />
          {eyebrow}
        </div>
      )}
      <h1 className={`hero-title${isHome ? " hero-title-home" : ""}`}>
        {title}
        {titleAccent && (
          <>
            <br />
            <span className="hero-title-accent">{titleAccent}</span>
          </>
        )}
      </h1>
      {subtitle && (
        <p className={`hero-subtitle${isHome ? " hero-subtitle-home" : ""}`}>
          {subtitle}
        </p>
      )}
    </header>
  );
}
