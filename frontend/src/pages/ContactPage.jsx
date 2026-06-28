import { useState } from "react";
import Footer from "../components/Footer";
import { submitContactForm, ApiError } from "../services/api";
import "../styles/layout.css";
import "./ContactPage.css";

const STATUS = {
  IDLE: "idle",
  SUBMITTING: "submitting",
  SUCCESS: "success",
  ERROR: "error",
  INCOMPLETE: "incomplete",
};

export default function ContactPage() {
  const [form, setForm] = useState({ name: "", email: "", message: "" });
  const [status, setStatus] = useState(STATUS.IDLE);
  const [errorMessage, setErrorMessage] = useState("");

  function updateField(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleSubmit(e) {
    e.preventDefault();

    if (!form.name || !form.email || !form.message) {
      setStatus(STATUS.INCOMPLETE);
      return;
    }

    setStatus(STATUS.SUBMITTING);
    try {
      await submitContactForm(form);
      setStatus(STATUS.SUCCESS);
      setForm({ name: "", email: "", message: "" });
    } catch (err) {
      setErrorMessage(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
      setStatus(STATUS.ERROR);
    }
  }

  return (
    <>
      <div className="page-bg-contact" />
      <div className="contact-hero">
        <div className="hero-eyebrow">◈ &nbsp; Get in Touch</div>
        <h1 className="hero-title">Let's Connect</h1>
        <p className="hero-subtitle" style={{ margin: "0 auto" }}>
          Have feedback, ideas, or just want to say hi? Drop a message below.
        </p>
      </div>

      <div className="contact-form-wrap">
        {status === STATUS.SUCCESS && (
          <div className="contact-success">Message sent. I'll get back to you soon.</div>
        )}
        {status === STATUS.ERROR && <div className="error-banner">{errorMessage}</div>}
        {status === STATUS.INCOMPLETE && (
          <div className="contact-warning">Please fill in all fields before submitting.</div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="contact-field">
            <label htmlFor="contact-name">Name</label>
            <input
              id="contact-name"
              type="text"
              placeholder="Your name"
              value={form.name}
              onChange={(e) => updateField("name", e.target.value)}
            />
          </div>
          <div className="contact-field">
            <label htmlFor="contact-email">Email</label>
            <input
              id="contact-email"
              type="email"
              placeholder="your@email.com"
              value={form.email}
              onChange={(e) => updateField("email", e.target.value)}
            />
          </div>
          <div className="contact-field">
            <label htmlFor="contact-message">Message</label>
            <textarea
              id="contact-message"
              placeholder="Share your thoughts..."
              value={form.message}
              onChange={(e) => updateField("message", e.target.value)}
            />
          </div>
          <button className="contact-submit" type="submit" disabled={status === STATUS.SUBMITTING}>
            {status === STATUS.SUBMITTING ? "Sending…" : "Send Message →"}
          </button>
        </form>
      </div>

      <Footer />
    </>
  );
}
