export default function Bodycam() {
  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Bodycam review</h1>
          <p className="page-sub">Review an officer's recording: a timeline of key moments, a transcript, and a report to export.</p>
        </div>
      </div>
      <section className="card card-body" style={{ maxWidth: 820 }}>
        <p style={{ marginTop: 0 }}>
          This workspace is next. It will use the same AI pipeline as Live CCTV, run over a whole recording:
        </p>
        <ul style={{ color: "#cfdbe3", lineHeight: 1.8 }}>
          <li>Upload or select a recording → key moments marked on a timeline (arrest, use of force, weapon, raised voices)</li>
          <li>Speech transcript in English, Hindi and Kannada, searchable</li>
          <li>Ask questions about the recording in the analyst chat</li>
          <li>Export a report; bystander faces and number plates blurred</li>
        </ul>
        <p className="note">Not built yet.</p>
      </section>
    </main>
  );
}
