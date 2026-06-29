import { useState } from "react";
import { MessageCircle, Sparkles } from "lucide-react";
import Hero from "../components/Hero";
import Footer from "../components/Footer";
import "../styles/layout.css";
import "./ContactPage.css";

const STATUS = { IDLE: "idle", SUCCESS: "success", INCOMPLETE: "incomplete" };
const WHATSAPP_NUMBER = "919004001598";

export default function ContactPage() {
  const [form, setForm] = useState({ name: "", email: "", message: "" });
  const [status, setStatus] = useState(STATUS.IDLE);

  function updateField(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!form.name || !form.email || !form.message) {
      setStatus(STATUS.INCOMPLETE);
      return;
    }
    const text = `Hi, I'm ${form.name} (${form.email}).\n\n${form.message}`;
    window.open(`https://wa.me/${WHATSAPP_NUMBER}?text=${encodeURIComponent(text)}`, "_blank", "noopener,noreferrer");
    setStatus(STATUS.SUCCESS);
    setForm({ name: "", email: "", message: "" });
  }

  return (
    <>
      <div className="page-bg-contact" />
      <Hero
        eyebrow="Get in Touch"
        title="Let's"
        titleAccent="Connect."
        subtitle="Have feedback, ideas, or just want to say hi? Drop a message below."
      />

      <div className="contact-form-wrap">
        {status === STATUS.SUCCESS && (
          <div className="contact-success">WhatsApp opened with your message. Talk soon!</div>
        )}
        {status === STATUS.INCOMPLETE && (
          <div className="contact-warning">Please fill in all fields before submitting.</div>
        )}
        <form onSubmit={handleSubmit}>
          <div className="contact-field">
            <label htmlFor="contact-name">Name</label>
            <input id="contact-name" type="text" placeholder="Your name"
              value={form.name} onChange={(e) => updateField("name", e.target.value)} />
          </div>
          <div className="contact-field">
            <label htmlFor="contact-email">Email</label>
            <input id="contact-email" type="email" placeholder="your@email.com"
              value={form.email} onChange={(e) => updateField("email", e.target.value)} />
          </div>
          <div className="contact-field">
            <label htmlFor="contact-message">Message</label>
            <textarea id="contact-message" placeholder="Share your thoughts..."
              value={form.message} onChange={(e) => updateField("message", e.target.value)} />
          </div>
          <button className="contact-submit" type="submit">
            <MessageCircle size={16} strokeWidth={2} /> Send via WhatsApp
          </button>
        </form>
      </div>
      <Footer />
    </>
  );
}
