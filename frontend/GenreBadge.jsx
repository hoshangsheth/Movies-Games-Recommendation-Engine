import { useEffect } from "react";
import "./Modal.css";

/**
 * Generic dialog shell, equivalent to the original's
 * `@st.dialog(..., width="large")` decorator. Closes on Escape or
 * backdrop click, matching standard modal expectations that Streamlit's
 * dialog also provided.
 */
export default function Modal({ onClose, children }) {
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="Close">
          ✕
        </button>
        {children}
      </div>
    </div>
  );
}
