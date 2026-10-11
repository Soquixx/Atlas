import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity, ArrowDownRight, ArrowUpRight, Bell, BookOpenCheck, BrainCircuit, MessageCircle,
  Building2, Check, ChevronDown, ChevronLeft, ChevronRight, CircleHelp,
  ClipboardList, Download, FileBarChart2, Filter, GraduationCap, LayoutDashboard,
  LoaderCircle, RefreshCw, Search, ShieldAlert, Sparkles, Target, Users,
  X, Database, CircleCheck, AlertTriangle, Lightbulb, SearchCheck, Cable, Upload, CheckCircle2, Clock3, Route, SearchCode, Workflow, ClipboardCheck, CircleGauge, TrendingDown, Stethoscope, ListChecks, Handshake,
} from 'lucide-react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const LIME = '#43E0C0';
const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'students', label: 'Student explorer', icon: Users },
  { id: 'riskpatterns', label: 'Risk patterns', icon: TrendingDown },
  { id: 'explanations', label: 'Risk explanations', icon: SearchCode },
  { id: 'actions', label: 'Data-driven actions', icon: ListChecks },
  { id: 'interventions', label: 'Interventions', icon: Workflow },
  { id: 'segments', label: 'Segments & blindspots', icon: Target },
  { id: 'sources', label: 'Data sources', icon: Cable },
  { id: 'quality', label: 'Data quality', icon: Database },
  { id: 'reports', label: 'Decision reports', icon: FileBarChart2 },
  { id: 'askatlas', label: 'Ask ATLAS', icon: MessageCircle },
] ;
const ROLE_PAGES = {
  'Institution admin': ['overview','students','riskpatterns','explanations','actions','interventions','segments','sources','quality','reports','askatlas'],
  'Department head': ['overview','students','riskpatterns','explanations','actions','interventions','segments','reports','askatlas'],
  'Faculty / mentor': ['students','explanations','actions','interventions','askatlas'],
  'Placement officer': ['overview','students','riskpatterns','explanations','actions','interventions','segments','reports','askatlas'],
};

async function getJson(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch { /* use status */ }
    throw new Error(detail);
  }
  return response.json();
}
const nfmt = (n) => Number(n ?? 0).toLocaleString('en-IN');
const one = (n) => Number(n ?? 0).toFixed(1);
const pct = (n) => `${one(n)}%`;
function analyticsDeltaText(result) {
  const before = result?.analytics_before;
  const after = result?.analytics_after;
  if (!before || !after) return '';
  const pairs = [
    ['Student profiles', before.student_count, after.student_count],
    ['Average LMS activity', before.average_lms_activity, after.average_lms_activity],
    ['Average engagement', before.average_engagement_score, after.average_engagement_score],
    ['Placement readiness', before.average_placement_readiness_score, after.average_placement_readiness_score],
    ['High placement-risk flags', before.high_placement_risk_count, after.high_placement_risk_count],
    ['Average success score', before.average_success_score, after.average_success_score],
  ];
  const fmt = (v) => v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toLocaleString('en-IN', { maximumFractionDigits: 1 });
  const deltas = pairs.map(([label, oldValue, newValue]) => `${label}: ${fmt(oldValue)} → ${fmt(newValue)}`);
  const changed = pairs.some(([, oldValue, newValue]) => oldValue != null && newValue != null && Math.abs(Number(oldValue) - Number(newValue)) > 0.05);
  if (!changed) deltas.push('Headline metrics unchanged; check whether the file differs from the active source and whether its Student_IDs match');
  return deltas.join(' · ');
}

function RiskPill({ level }) {
  const key = String(level || 'Unknown').toLowerCase();
  return <span className={`risk-pill ${key.replaceAll(' ', '-')}`}>{level || 'Unknown'}</span>;
}
function ScoreRing({ value = 0, size = 58 }) {
  const v = Math.max(0, Math.min(100, Number(value || 0)));
  return <div className="score-ring" style={{ '--score': `${v * 3.6}deg`, width: size, height: size }}><span>{Math.round(v)}</span></div>;
}
function CardHeader({ title, subtitle, action }) {
  return <div className="card-head"><div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>{action}</div>;
}
function Loading() { return <div className="loading"><LoaderCircle className="spin" size={24} /> Loading analytics…</div>; }
function Empty({ title = 'No data to display', text = 'Try changing your filters.' }) { return <div className="empty"><SearchCheck size={26} /><b>{title}</b><span>{text}</span></div>; }

function Dashboard({ workspace }) {
  const allowedPages = ROLE_PAGES[workspace?.role] || ROLE_PAGES['Institution admin'];
  const visibleNav = NAV.filter(item => allowedPages.includes(item.id));
  const defaultPage = allowedPages.includes('overview') ? 'overview' : allowedPages[0];
  const [page, setPage] = useState(defaultPage);
  const [overview, setOverview] = useState(null);
  const [branches, setBranches] = useState(['All']);
  const [segments, setSegments] = useState([]);
  const [discoveredPatterns, setDiscoveredPatterns] = useState([]);
  const [riskRows, setRiskRows] = useState([]);
  const [riskTotal, setRiskTotal] = useState(0);
  const [interventionPlans, setInterventionPlans] = useState(() => {
    try { const saved = localStorage.getItem(`atlas-interventions-${workspace?.institution || 'demo'}`); if (saved) return JSON.parse(saved); } catch {}
    return [
    { id: 'ACT-001', title: 'Course-level review: foundational programming', scope: 'CSE · Year 2', rationale: 'Review course-level marks, backlog concentration and recent engagement signals before choosing support.', owner: 'Department coordinator', status: 'Proposed', students: 42 },
    { id: 'ACT-002', title: 'Placement-readiness practice cohort', scope: 'Cross-branch · Final year', rationale: 'Prioritize coding practice and mock-interview review for students whose placement-readiness indicators are low.', owner: 'Placement cell', status: 'Planned', students: 68 },
    { id: 'ACT-003', title: 'Attendance and submission follow-up', scope: 'Multiple cohorts', rationale: 'Verify attendance and assignment records with faculty before contacting students; missing records can distort signals.', owner: 'Class advisors', status: 'Proposed', students: 31 },
    ];
  });
  useEffect(() => { try { localStorage.setItem(`atlas-interventions-${workspace?.institution || 'demo'}`, JSON.stringify(interventionPlans)); } catch {} }, [interventionPlans, workspace?.institution]);
  const [quality, setQuality] = useState(null);
  const [sources, setSources] = useState([]);
  const [sourceNote, setSourceNote] = useState('');
  const [syncingSource, setSyncingSource] = useState('');
  const [students, setStudents] = useState({ items: [], total: 0, page: 1, page_size: 25 });
  const [filters, setFilters] = useState({ search: '', branch: 'All', segment: 'All', risk: 'All', sort_by: 'student_success_score', sort_dir: 'asc' });
  const [tablePage, setTablePage] = useState(1);
  const [selectedId, setSelectedId] = useState(null);
  const [studentDetail, setStudentDetail] = useState(null);
  const [report, setReport] = useState(null);
  const [reportBranch, setReportBranch] = useState('All');
  const [reportType, setReportType] = useState('risk_snapshot');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState(null);
  const [atlasQuestion, setAtlasQuestion] = useState('Which current patterns deserve attention, and what evidence supports them?');
  const [atlasAnswer, setAtlasAnswer] = useState(null);
  const [atlasBusy, setAtlasBusy] = useState(false);

  const loadCore = useCallback(async () => {
    setError('');
    try {
      const [o, b, s, p] = await Promise.all([getJson('/api/overview'), getJson('/api/branches'), getJson('/api/segments'), getJson('/api/patterns')]);
      setOverview(o); setBranches(b.items); setSegments(s.items); setDiscoveredPatterns(p.items || []); setLastUpdated(new Date());
    } catch (e) { setError(`ATLAS couldn't load the workspace data. Check that the workspace is running. Details: ${e.message}`); }
  }, []);
  useEffect(() => { loadCore(); }, [loadCore]);

  const loadStudents = useCallback(async () => {
    setBusy(true);
    try {
      const params = new URLSearchParams({ ...filters, page: String(tablePage), page_size: '25' });
      setStudents(await getJson(`/api/students?${params.toString()}`));
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }, [filters, tablePage]);
  useEffect(() => { if (page === 'students') loadStudents(); }, [page, loadStudents]);

  useEffect(() => {
    if (['riskpatterns', 'explanations', 'actions', 'interventions'].includes(page)) {
      getJson('/api/students?risk=Any%20High&page=1&page_size=100&sort_by=student_success_score&sort_dir=asc')
        .then(d => { setRiskRows(d.items || []); setRiskTotal(d.total || 0); })
        .catch(e => setError(e.message));
      if (!segments.length) getJson('/api/segments').then(d => setSegments(d.items || [])).catch(e => setError(e.message));
    }
  }, [page]);

  useEffect(() => {
    if (page === 'quality' && !quality) getJson('/api/data-quality').then(setQuality).catch(e => setError(e.message));
    if (page === 'sources') getJson('/api/sources').then(d => setSources(d.items)).catch(e => setError(e.message));
  }, [page, quality]);

  useEffect(() => {
    if (!selectedId) { setStudentDetail(null); return; }
    getJson(`/api/students/${encodeURIComponent(selectedId)}`).then(setStudentDetail).catch(e => setError(e.message));
  }, [selectedId]);

  const generateReport = async () => {
    setBusy(true); setError('');
    try { setReport(await getJson('/api/reports/generate', { method: 'POST', body: JSON.stringify({ report_type: reportType, branch: reportBranch }) })); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };

  const goStudents = (risk = 'All', segment = 'All', branch = 'All') => {
    setFilters(f => ({ ...f, risk, segment, branch, search: '' })); setTablePage(1); setPage('students');
  };

  const renderOverview = () => {
    if (!overview) return <Loading />;
    const kpis = [
      { label: 'Students analyzed', value: nfmt(overview.student_count), sub: 'Joined student profiles', icon: Users, tone: 'black', suffix: '↑' },
      { label: 'Average success score', value: one(overview.average_success_score), sub: 'Composite score / 100', icon: Activity, tone: 'lime', score: overview.average_success_score },
      { label: 'Academic risk', value: nfmt(overview.high_academic_risk_count), sub: `${nfmt(overview.moderate_academic_risk_count)} moderate · rules-based`, icon: BookOpenCheck, tone: 'red', click: () => goStudents('Academic High') },
      { label: 'Placement risk', value: nfmt(overview.high_placement_risk_count), sub: 'Estimated from demo data', icon: ShieldAlert, tone: 'purple', click: () => goStudents('Placement High') },
    ];
    const riskData = [
      { name: 'Academic', high: overview.high_academic_risk_count, moderate: overview.moderate_academic_risk_count },
      { name: 'Placement', high: overview.high_placement_risk_count, moderate: overview.moderate_placement_risk_count },
    ];
    return <>
      <div className="page-intro"><div><div className="eyebrow">CAMPUS INTELLIGENCE / OVERVIEW</div><h1>Good morning. <span>Let's look closer.</span></h1><p>See where students are thriving, where support may be needed, and what deserves attention next.</p></div><button className="btn btn-secondary" onClick={loadCore}><RefreshCw size={15} /> Refresh data</button></div>
      <div className="alert-strip"><div className="alert-icon"><Database size={17} /></div><div><strong>Demo environment · Synthetic data</strong><span>20,000 generated student profiles across connected campus data sources. Insights are for demonstration only, not for real student decisions.</span></div><span className="demo-tag">DEMO DATA</span></div>
      <div className="kpi-grid">{kpis.map((k, i) => <button key={k.label} className={`kpi-card ${k.tone} ${k.click ? 'clickable' : ''}`} onClick={k.click || undefined}><div className="kpi-top"><span>{k.label}</span><k.icon size={18} /></div><div className="kpi-value">{k.value}</div><div className="kpi-bottom"><span>{k.sub}</span>{i === 0 ? <span className="mini-stamp">LIVE DATASET</span> : k.click ? <ChevronRight size={16} /> : null}</div></button>)}</div>
      <div className="score-strip"><div className="score-strip-title"><div className="icon-bubble"><BrainCircuit size={19} /></div><div><b>Student Success Index</b><span>Three perspectives, one view — without hiding what drives the score.</span></div></div><div className="score-metric"><span>Academic</span><b>{one(overview.average_academic_score)}</b><div className="mini-progress"><i style={{ width: `${overview.average_academic_score}%` }} /></div></div><div className="score-metric"><span>Engagement</span><b>{one(overview.average_engagement_score)}</b><div className="mini-progress"><i style={{ width: `${overview.average_engagement_score}%` }} /></div></div><div className="score-metric"><span>Placement readiness</span><b>{one(overview.average_placement_readiness_score)}</b><div className="mini-progress"><i style={{ width: `${overview.average_placement_readiness_score}%` }} /></div></div><button className="arrow-button" onClick={() => setPage('segments')} aria-label="View segments"><ChevronRight /></button></div>
      <div className="chart-grid">
        <section className="panel chart-panel wide"><CardHeader title="Activity, week over week" subtitle="Average LMS activity index across all connected student records" action={<span className="period-pill">16 WEEKS</span>} />
          <div className="chart-body"><ResponsiveContainer width="100%" height={260}><AreaChart data={overview.weekly_trend} margin={{ top: 12, right: 14, left: -18, bottom: 0 }}><defs><linearGradient id="limeFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={LIME} stopOpacity={0.55} /><stop offset="100%" stopColor={LIME} stopOpacity={0.02} /></linearGradient></defs><CartesianGrid vertical={false} stroke="#e8e9e2" strokeDasharray="3 4" /><XAxis dataKey="week" axisLine={false} tickLine={false} tick={{ fill: '#888b82', fontSize: 11 }} dy={8} /><YAxis axisLine={false} tickLine={false} tick={{ fill: '#888b82', fontSize: 11 }} domain={[0, 100]} /><Tooltip contentStyle={{ border: '1px solid #e6e8de', borderRadius: 12, fontSize: 12 }} /><Area type="monotone" dataKey="average_activity" name="Activity index" stroke="#191b16" strokeWidth={2.5} fill="url(#limeFill)" activeDot={{ r: 5, fill: LIME, stroke: '#191b16', strokeWidth: 2 }} /></AreaChart></ResponsiveContainer></div>
          <div className="chart-foot"><span><i className="legend-dot lime-dot" /> Average activity index</span><span className="muted">Changes are descriptive, not causal.</span></div>
        </section>
        <section className="panel chart-panel"><CardHeader title="Risk landscape" subtitle="Students flagged for review" action={<button className="link-btn" onClick={() => goStudents('Any High')}>View list <ChevronRight size={14} /></button>} />
          <div className="chart-body"><ResponsiveContainer width="100%" height={250}><BarChart data={riskData} barGap={9} margin={{ top: 18, right: 2, left: -18, bottom: 0 }}><CartesianGrid vertical={false} stroke="#e8e9e2" strokeDasharray="3 4" /><XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#66695f', fontSize: 12 }} dy={8} /><YAxis axisLine={false} tickLine={false} tick={{ fill: '#888b82', fontSize: 11 }} /><Tooltip contentStyle={{ border: '1px solid #e6e8de', borderRadius: 12, fontSize: 12 }} /><Bar dataKey="high" name="High priority" fill="#f16f62" radius={[5, 5, 0, 0]} maxBarSize={34} /><Bar dataKey="moderate" name="Moderate" fill={LIME} radius={[5, 5, 0, 0]} maxBarSize={34} /></BarChart></ResponsiveContainer></div>
          <div className="chart-foot"><span><i className="legend-dot red-dot" /> High priority</span><span><i className="legend-dot lime-dot" /> Moderate review</span></div>
        </section>
      </div>
      <section className="panel branch-panel"><CardHeader title="Department snapshot" subtitle="Compare score averages and risk counts across branches" action={<button className="link-btn" onClick={() => setPage('students')}>Explore students <ChevronRight size={14} /></button>} />
        <div className="branch-table"><div className="branch-table-head"><span>Branch</span><span>Students</span><span>Average success</span><span>Academic high risk</span><span>Placement high risk</span></div>{overview.branches.map((b) => <button key={b.branch} className="branch-row" onClick={() => goStudents('All', 'All', b.branch)}><span className="branch-name"><i>{b.branch.slice(0, 2).toUpperCase()}</i><b>{b.branch}</b></span><span>{nfmt(b.students)}</span><span><div className="score-inline"><div className="mini-progress"><i style={{ width: `${b.average_success_score}%` }} /></div><b>{one(b.average_success_score)}</b></div></span><span><RiskPill level={b.academic_risk_count > 0 ? 'Review' : 'Low'} /> <small>{nfmt(b.academic_risk_count)}</small></span><span><RiskPill level={b.placement_risk_count > 0 ? 'Review' : 'Low'} /> <small>{nfmt(b.placement_risk_count)}</small></span></button>)}</div>
      </section>
      <div className="bottom-grid"><section className="panel mini-panel"><div className="mini-icon lime-bg"><Lightbulb size={18} /></div><div><div className="eyebrow">SIGNAL TO INVESTIGATE</div><h3>Find the blindspot behind the average</h3><p>Students can have strong academic results and still show a placement-readiness gap. Inspect segments to find where outcomes diverge.</p><button className="btn btn-dark" onClick={() => setPage('segments')}>Explore segments <ChevronRight size={15} /></button></div></section><section className="panel model-panel"><div className="mini-icon violet-bg"><BrainCircuit size={18} /></div><div><div className="eyebrow">HOW TO READ RISK</div><h3>Risk is a signal, not a verdict</h3><p>Placement risk is an estimate from recorded indicators. These synthetic demo records illustrate the workflow, not real-world placement outcomes.</p><button className="link-btn" onClick={() => setPage('quality')}>Inspect data quality <ChevronRight size={14} /></button></div></section></div>
    </>;
  };

  const renderStudents = () => <>
    <div className="page-intro"><div><div className="eyebrow">CAMPUS INTELLIGENCE / STUDENTS</div><h1>Student <span>explorer.</span></h1><p>Search the unified view, inspect score drivers and identify students who may benefit from a closer review.</p></div><button className="btn btn-dark" onClick={() => { const link = `${API}/api/students?branch=${encodeURIComponent(filters.branch)}&page_size=100`; window.open(link, '_blank', 'noopener,noreferrer'); }}><Download size={15} /> Export student data</button></div>
    <section className="panel student-explorer"><div className="filterbar"><label className="search-box"><Search size={16} /><input placeholder="Search student ID…" value={filters.search} onChange={e => { setFilters(f => ({ ...f, search: e.target.value })); setTablePage(1); }} /></label><label className="filter-select"><Building2 size={15} /><select value={filters.branch} onChange={e => { setFilters(f => ({ ...f, branch: e.target.value })); setTablePage(1); }}>{branches.map(b => <option key={b}>{b}</option>)}</select></label><label className="filter-select"><Filter size={15} /><select value={filters.risk} onChange={e => { setFilters(f => ({ ...f, risk: e.target.value })); setTablePage(1); }}>{['All', 'Any High', 'Academic High', 'Academic Moderate', 'Placement High', 'Placement Moderate'].map(v => <option key={v}>{v}</option>)}</select></label><label className="filter-select"><Target size={15} /><select value={filters.segment} onChange={e => { setFilters(f => ({ ...f, segment: e.target.value })); setTablePage(1); }}><option>All</option>{segments.map(s => <option key={s.segment}>{s.segment}</option>)}</select></label></div>
      {busy ? <Loading /> : students.items.length === 0 ? <Empty title="No students match these filters" /> : <><div className="table-scroll"><table className="data-table"><thead><tr><th>Student</th><th>Success score</th><th>Academics</th><th>Placement readiness</th><th>Academic risk</th><th>Placement risk</th><th>Segment</th><th></th></tr></thead><tbody>{students.items.map(s => <tr key={s.student_id} onClick={() => setSelectedId(s.student_id)}><td><div className="student-cell"><span className="avatar">{s.student_id.slice(-2)}</span><span><b>{s.student_id}</b><small>{s.branch} · Avg. marks {one(s.average_course_marks)}</small></span></div></td><td><div className="score-inline"><ScoreRing value={s.success_score} size={38} /><b>{one(s.success_score)}</b></div></td><td><div className="simple-score"><b>{one(s.academic_score)}</b><div className="mini-progress"><i style={{ width: `${s.academic_score}%` }} /></div></div></td><td><div className="simple-score"><b>{one(s.placement_readiness_score)}</b><div className="mini-progress"><i style={{ width: `${s.placement_readiness_score}%` }} /></div></div></td><td><RiskPill level={s.academic_risk} /></td><td><RiskPill level={s.placement_risk} /></td><td><span className="segment-text">{s.segment}</span></td><td><ChevronRight size={16} color="#92958b" /></td></tr>)}</tbody></table></div><div className="table-paging"><span>Showing {students.total ? (tablePage - 1) * 25 + 1 : 0}–{Math.min(tablePage * 25, students.total)} of {nfmt(students.total)} students</span><div><button disabled={tablePage <= 1} onClick={() => setTablePage(p => Math.max(1, p - 1))}><ChevronLeft size={16} /></button><b>{tablePage}</b><button disabled={tablePage * 25 >= students.total} onClick={() => setTablePage(p => p + 1)}><ChevronRight size={16} /></button></div></div></>}
    </section>
    {selectedId && <div className="modal-backdrop" onClick={() => setSelectedId(null)}><div className="student-modal" onClick={e => e.stopPropagation()}><button className="modal-close" onClick={() => setSelectedId(null)}><X size={18} /></button>{!studentDetail ? <Loading /> : <StudentProfile detail={studentDetail} />}</div></div>}
  </>;

  const renderSegments = () => <>
    <div className="page-intro"><div><div className="eyebrow">CAMPUS INTELLIGENCE / SEGMENTATION</div><h1>Signals behind <span>the scores.</span></h1><p>Explore meaningful student profiles where one composite score would hide the more important story.</p></div></div>
    <div className="segment-hero"><div><div className="eyebrow">SIGNATURE CAPABILITY</div><h2>Success Blindspot Engine</h2><p>Surface gaps between academics, engagement and placement preparation—then inspect the evidence before deciding what to do next.</p><div className="hero-tags"><span><Check size={13} /> Score explanations</span><span><Check size={13} /> Cohort patterns</span><span><Check size={13} /> Evidence-first</span></div></div><div className="venn"><div className="venn-circle one">Academic</div><div className="venn-circle two">Engagement</div><div className="venn-circle three">Placement</div><div className="venn-core">SUCCESS<br/>SIGNALS</div></div></div>
    <div className="segment-grid">{segments.map((s, i) => <button className={`segment-card seg-${i % 5}`} key={s.segment} onClick={() => goStudents('All', s.segment)}><div className="segment-card-top"><span className="segment-no">{String(i + 1).padStart(2, '0')}</span><ChevronRight size={17} /></div><h3>{s.segment}</h3><div className="segment-count">{nfmt(s.student_count)} <span>students</span></div><div className="segment-score-row"><span>Avg. success score</span><b>{one(s.average_success_score)}</b></div><div className="mini-progress"><i style={{ width: `${s.average_success_score}%` }} /></div><div className="segment-breakdown"><span>Academic {one(s.average_academic_score)}</span><span>Placement {one(s.average_placement_score)}</span></div><div className="segment-open">Inspect this segment <ChevronRight size={14} /></div></button>)}</div>
    <section className="panel explain-panel"><div className="explain-icon"><BrainCircuit size={21} /></div><div className="explain-copy"><div className="eyebrow">WHY THIS MATTERS</div><h3>Explainable before actionable</h3><p>Every flagged student has explicit reasons drawn from source records—such as attendance below the review threshold, low course marks, a backlog, a falling LMS activity trend, weak mock-coding results or a recorded support request. These are prompts for review, not conclusions about a student.</p></div><button className="btn btn-dark" onClick={() => goStudents('Any High')}>Review flagged students <ChevronRight size={15} /></button></section>
  </>;

  const riskPatternRows = useMemo(() => {
    const all = segments || [];
    return all.map((s, i) => {
      const academic = Number(s.average_academic_score || 0);
      const placement = Number(s.average_placement_score || 0);
      const gap = academic - placement;
      const riskCount = Number(s.high_academic_risk || 0) + Number(s.high_placement_risk || 0);
      let pattern = 'Mixed signals';
      let explanation = 'Scores vary across dimensions; review individual records before choosing support.';
      let action = 'Review the segment and verify source completeness.';
      if (academic < 60) { pattern = 'Academic performance pressure'; explanation = 'The segment has a low average academic score and may include students with course-level difficulties.'; action = 'Review course marks, backlogs and assessment completion with the course coordinator.'; }
      else if (placement < 60 || gap >= 12) { pattern = 'Placement-readiness gap'; explanation = 'Academic performance is stronger than placement readiness, suggesting a possible preparation gap rather than a purely academic one.'; action = 'Inspect coding, aptitude and mock-interview evidence; offer a targeted practice cycle if verified.'; }
      else if (Number(s.high_academic_risk || 0) > Number(s.high_placement_risk || 0)) { pattern = 'Academic risk concentration'; explanation = 'Rule-triggered academic flags are more common than placement flags in this segment.'; action = 'Validate course-level signals and prioritize a faculty review.'; }
      return { ...s, index: i, pattern, explanation, action, gap, riskCount };
    }).sort((a,b) => b.riskCount-a.riskCount);
  }, [segments]);

  const updateIntervention = (id, status) => setInterventionPlans(prev => prev.map(x => x.id === id ? { ...x, status } : x));
  const createActionPlan = (title, scope, rationale, students = 0, owner = 'Department coordinator') => {
    const lower = `${title} ${rationale}`.toLowerCase();
    const intervention = lower.includes('placement') || lower.includes('coding') || lower.includes('aptitude')
      ? { kind: 'Placement practice cohort', targetMetric: 'Placement readiness score', followUp: 'Repeat the relevant practice assessment after the support cycle.' }
      : lower.includes('attendance') || lower.includes('lms') || lower.includes('submission')
        ? { kind: 'Engagement follow-up', targetMetric: 'Attendance / on-time submission rate', followUp: 'Compare the same indicator at the next reporting point.' }
        : { kind: 'Academic support review', targetMetric: 'Course marks and pass rate', followUp: 'Compare the next assessment result with the recorded baseline.' };
    const newPlan = { id: `ACT-${String(Date.now()).slice(-5)}`, title, scope, rationale, students, owner, status: 'Proposed', ...intervention, createdAt: new Date().toISOString() };
    setInterventionPlans(prev => [newPlan, ...prev]);
  };

  const renderRiskPatterns = () => <>
    <div className="page-intro"><div><div className="eyebrow">ATLAS / PATTERN DETECTION</div><h1>Risks have <span>patterns.</span></h1><p>Move from isolated flags to recurring cohort-level signals. Patterns are recalculated from the active student records and should be verified before action.</p></div><button className="btn btn-secondary" onClick={() => setPage('explanations')}><SearchCode size={15}/> Inspect explanations</button></div>
    <div className="lagoon-banner"><div className="lagoon-icon"><Route size={23}/></div><div><b>Student journey pattern explorer</b><p>Compare academic performance, placement readiness and rule-triggered risk counts to surface where support may be needed—not to infer causation.</p></div><span className="demo-tag">SYNTHETIC DATA</span></div>
    <div className="kpi-grid pattern-kpis"><div className="kpi-card"><div className="kpi-top">High-risk students in returned sample <ShieldAlert size={18}/></div><div className="kpi-value">{nfmt(riskTotal)}</div><div className="kpi-bottom">Students flagged high · academic or placement</div></div><div className="kpi-card"><div className="kpi-top">Segments surfaced <Target size={18}/></div><div className="kpi-value">{nfmt(riskPatternRows.length)}</div><div className="kpi-bottom">Derived from current segment summaries</div></div><div className="kpi-card"><div className="kpi-top">Priority pattern <TrendingDown size={18}/></div><div className="kpi-value">{riskPatternRows[0]?.pattern || '—'}</div><div className="kpi-bottom">Sorted by combined high-risk counts</div></div><div className="kpi-card"><div className="kpi-top">Evidence standard <ShieldAlert size={18}/></div><div className="kpi-value">Review first</div><div className="kpi-bottom">Signals are not causal explanations</div></div></div>
    <section className="panel" style={{padding:'22px',marginBottom:'18px'}}><div className="card-head"><div><h3>Findings calculated from the active data</h3><p>Course averages, cohort differences and LMS trajectories are recalculated from currently active source files.</p></div><span className="period-pill">{discoveredPatterns.length} FINDINGS</span></div>{discoveredPatterns.length ? <div className="pattern-grid">{discoveredPatterns.slice(0,6).map(item=><article className="panel pattern-card" key={item.id}><div className="eyebrow">{item.category}</div><h3>{item.title}</h3><p>{item.interpretation}</p><div className="pattern-metrics"><div><span>Students</span><b>{nfmt(item.student_count)}</b></div>{Object.entries(item.evidence || {}).slice(0,3).map(([k,v])=><div key={k}><span>{k.replaceAll('_',' ')}</span><b>{typeof v === 'number' ? one(v) : String(v)}</b></div>)}</div><div className="pattern-action"><Lightbulb size={16}/><span>{item.suggested_action}</span></div></article>)}</div> : <Empty title="No qualifying patterns found" text="Check data coverage or upload a new source to recalculate."/>}</section>
    <div className="pattern-grid">{riskPatternRows.map((s,i)=><section className="panel pattern-card" key={s.segment}><div className="pattern-card-top"><span className="pattern-index">0{i+1}</span><RiskPill level={s.riskCount > 0 ? 'Review' : 'Low'}/></div><div className="eyebrow">{s.pattern.toUpperCase()}</div><h3>{s.segment}</h3><p>{s.explanation}</p><div className="pattern-metrics"><div><span>Students</span><b>{nfmt(s.student_count)}</b></div><div><span>Academic avg.</span><b>{one(s.average_academic_score)}</b></div><div><span>Placement avg.</span><b>{one(s.average_placement_score)}</b></div><div><span>High-risk flags*</span><b>{nfmt(s.riskCount)}</b></div></div><div className="pattern-gap"><span>Academic ↔ placement gap</span><b className={s.gap >= 12 ? 'risk-gap' : ''}>{s.gap > 0 ? '+' : ''}{one(s.gap)} pts</b></div><div className="pattern-action"><Lightbulb size={16}/><span>{s.action}</span></div><button className="btn btn-secondary" onClick={() => { setFilters(f=>({...f,segment:s.segment,risk:'All',search:''})); setPage('students'); }}>Explore this segment <ChevronRight size={15}/></button></section>)}</div>
    <p className="footnote">*The combined count can include a student flagged in both risk categories. Segment averages describe observed synthetic records and do not establish why a pattern occurred.</p>
  </>;

  const renderExplanations = () => <>
    <div className="page-intro"><div><div className="eyebrow">ATLAS / EXPLAINABILITY</div><h1>Show the <span>why.</span></h1><p>Every risk flag should be reviewable through the indicators that triggered it, the records available and the limits of the evidence.</p></div><button className="btn btn-secondary" onClick={() => setPage('students')}><Users size={15}/> Open student explorer</button></div>
    <div className="explain-layout"><section className="panel explain-method"><div className="method-symbol"><BrainCircuit size={24}/></div><div className="eyebrow">VISIBLE EVIDENCE + CONTEXT</div><h2>Risk explanation, not a black-box verdict.</h2><p>Academic risk uses recorded indicators and clear review rules. Placement risk is estimated from available student information. The current bundle is synthetic, so scores demonstrate the workflow—not real-world accuracy.</p><div className="method-checks"><div><CircleCheck size={16}/><span>Show the exact indicators behind each flag.</span></div><div><CircleCheck size={16}/><span>Keep observed facts separate from interpretations and recommendations.</span></div><div><CircleCheck size={16}/><span>Ask an authorized staff member to review context before acting.</span></div></div><button className="btn btn-dark" onClick={() => setPage('quality')}>Review data quality <ChevronRight size={15}/></button></section><section className="panel explanation-table"><CardHeader title="Students with high-risk flags" subtitle={`${nfmt(riskRows.length)} sample rows shown from ${nfmt(riskTotal)} matching profiles`} action={<span className="period-pill">CURRENT DATA</span>}/>{riskRows.length ? <div className="explanation-list">{riskRows.slice(0,18).map(st=><article className="explanation-row" key={st.student_id}><div className="explanation-person"><span className="avatar">{st.student_id.slice(-2)}</span><div><b>{st.student_id}</b><small>{st.branch} · {st.segment}</small></div></div><div className="explanation-risk"><RiskPill level={st.academic_risk}/><RiskPill level={st.placement_risk}/></div><div className="explanation-drivers">{(st.explanation_drivers || []).length ? st.explanation_drivers.slice(0,3).map((d,i)=><span key={i}><Activity size={13}/>{d}</span>) : <span>No configured driver returned; verify source completeness.</span>}</div><button className="link-btn" onClick={()=>setSelectedId(st.student_id)}>Inspect profile <ChevronRight size={14}/></button></article>)}</div> : <Loading/>}</section></div>
    {selectedId && <div className="modal-backdrop" onClick={() => setSelectedId(null)}><div className="student-modal" onClick={e => e.stopPropagation()}><button className="modal-close" onClick={() => setSelectedId(null)}><X size={18}/></button>{!studentDetail ? <Loading/> : <StudentProfile detail={studentDetail}/>}</div></div>}
  </>;

  const renderActions = () => <>
    <div className="page-intro"><div><div className="eyebrow">ATLAS / INSTITUTIONAL DECISION INTELLIGENCE</div><h1>Turn signals into <span>decisions.</span></h1><p>Translate observed patterns into proportionate, human-reviewed actions with a clear owner and a measurable follow-up.</p></div><button className="btn btn-dark" onClick={() => setPage('interventions')}><Workflow size={15}/> View intervention tracker</button></div>
    <div className="action-principle"><div><div className="eyebrow">DECISION FRAMEWORK</div><h2>Evidence → action → follow-up</h2><p>Actions below are suggested checks based on current indicators. They are not automatically assigned to students and should be approved by authorized staff.</p></div><div className="action-steps"><span><b>01</b> Verify</span><span><b>02</b> Prioritize</span><span><b>03</b> Measure</span></div></div>
    <div className="action-list">{discoveredPatterns.slice(0,6).map((s,i)=><article className="panel action-card" key={s.id}><div className="action-priority"><span className={`priority-dot p${i%3}`}/><small>REVIEW {String(i+1).padStart(2,'0')}</small><b>{nfmt(s.student_count)} in scope</b></div><div className="action-main"><div className="eyebrow">{s.category.toUpperCase()}</div><h3>{s.suggested_action}</h3><p><b>Observed basis:</b> {s.title}. {s.interpretation} <b>Evidence:</b> {JSON.stringify(s.evidence)}.</p><div className="action-meta"><span><Users size={14}/> Scope: {nfmt(s.student_count)} students</span><span><Handshake size={14}/> Suggested owner: {s.owner}</span><span><CircleGauge size={14}/> Measure: {s.measure}</span></div></div><button className="btn btn-secondary" onClick={()=>createActionPlan(s.suggested_action,s.scope,s.interpretation,s.student_count,s.owner)}>Add to action plan <ChevronRight size={15}/></button></article>)}</div><p className="footnote">These action suggestions are derived from current descriptive findings. Authorized staff must verify the evidence and approve support; no recommendation is a causal conclusion.</p>
  </>;

  const renderInterventions = () => <>
    <div className="page-intro"><div><div className="eyebrow">ATLAS / INTERVENTION WORKFLOW</div><h1>Make support <span>trackable.</span></h1><p>Move proposed actions through review and delivery, then record a follow-up measure. This prototype tracker is session-only and does not contact students.</p></div><button className="btn btn-secondary" onClick={()=>setPage('actions')}><ListChecks size={15}/> Find actions</button></div>
    <div className="kpi-grid"><div className="kpi-card"><div className="kpi-top">Proposed <ClipboardList size={18}/></div><div className="kpi-value">{interventionPlans.filter(x=>x.status==='Proposed').length}</div><div className="kpi-bottom">Awaiting human review</div></div><div className="kpi-card"><div className="kpi-top">Planned <Clock3 size={18}/></div><div className="kpi-value">{interventionPlans.filter(x=>x.status==='Planned').length}</div><div className="kpi-bottom">Reviewed, not yet delivered</div></div><div className="kpi-card"><div className="kpi-top">Delivered <CheckCircle2 size={18}/></div><div className="kpi-value">{interventionPlans.filter(x=>x.status==='Delivered').length}</div><div className="kpi-bottom">Marked delivered in this demo</div></div><div className="kpi-card"><div className="kpi-top">Follow-up due <Activity size={18}/></div><div className="kpi-value">{interventionPlans.filter(x=>x.status==='Delivered').length}</div><div className="kpi-bottom">Record an outcome before claiming impact</div></div></div>
    <section className="panel intervention-panel"><CardHeader title="Intervention register" subtitle="Suggested actions are editable workflow records; student-level decisions require authorized human review." action={<span className="period-pill">SESSION DEMO</span>}/><div className="intervention-table-head"><span>Action / scope</span><span>Owner</span><span>Students</span><span>Status</span><span>Workflow</span></div>{interventionPlans.map(item=><article className="intervention-row" key={item.id}><div><small className="action-id">{item.id}</small><b>{item.title}</b><span>{item.scope}</span><p>{item.rationale}</p>{item.targetMetric && <p className="caveat-text"><b>Measure:</b> {item.targetMetric} · <b>Follow-up:</b> {item.followUp}</p>}</div><div className="owner-cell">{item.owner}</div><div className="student-count-cell">{nfmt(item.students)}</div><div><span className={`workflow-status ${item.status.toLowerCase()}`}>{item.status}</span></div><div className="workflow-buttons">{item.status==='Proposed' && <button className="btn btn-secondary" onClick={()=>updateIntervention(item.id,'Planned')}>Approve plan</button>}{item.status==='Planned' && <button className="btn btn-teal" onClick={()=>updateIntervention(item.id,'Delivered')}>Mark delivered</button>}{item.status==='Delivered' && <button className="btn btn-secondary" onClick={()=>updateIntervention(item.id,'Follow-up due')}>Record follow-up due</button>}{item.status==='Follow-up due' && <button className="btn btn-secondary" onClick={()=>updateIntervention(item.id,'Reviewed')}>Mark reviewed</button>}{item.status==='Reviewed' && <span className="completed-label"><CheckCircle2 size={14}/> Reviewed</span>}</div></article>)}</section>
    <section className="panel followup-panel"><div className="followup-icon"><ClipboardCheck size={22}/></div><div><div className="eyebrow">OUTCOME DISCIPLINE</div><h3>Delivered is not the same as effective.</h3><p>For a real evaluation, record the intervention offered, accepted, delivered date, a pre-defined baseline indicator, a follow-up value and follow-up coverage. A before/after change is descriptive; it does not prove the intervention caused the change.</p></div></section>
  </>;

  const syncSource = async (filename) => {
    setSyncingSource(filename); setSourceNote(''); setError('');
    try {
      const result = await getJson(`/api/sources/${encodeURIComponent(filename)}/sync`, { method: 'POST' });
      const sameFile = result.source_content_changed === false;
      const status = result.activated_upload
        ? (sameFile ? 'The uploaded CSV is identical to the active source.' : 'Uploaded CSV activated; analytics recalculated.')
        : 'Active source checked and analytics refreshed.';
      const delta = analyticsDeltaText(result);
      const joinWarning = result.join_diagnostic?.warning ? ` ID matching: ${result.join_diagnostic.match_pct}% matched. ${result.join_diagnostic.warning}` : '';
      setSourceNote(`${filename}: ${status} ${nfmt(result.row_count)} rows; ${nfmt(result.student_count)} student profiles. ${delta}${joinWarning}`);
      setSources((await getJson('/api/sources')).items);
      setQuality(null); setReport(null); await loadCore();
      if (['riskpatterns', 'explanations', 'actions', 'interventions'].includes(page)) {
        const riskData = await getJson('/api/students?risk=Any%20High&page=1&page_size=100&sort_by=student_success_score&sort_dir=asc');
        setRiskRows(riskData.items || []); setRiskTotal(riskData.total || 0);
      }
      if (page === 'students') await loadStudents();
      if (selectedId) {
        const detail = await getJson(`/api/students/${encodeURIComponent(selectedId)}`).catch(() => null);
        setStudentDetail(detail);
      }
    } catch (e) { setError(typeof e.message === 'string' ? e.message : String(e)); } finally { setSyncingSource(''); }
  };
  const uploadSource = async (filename, file) => {
    if (!file) return; setSyncingSource(filename); setSourceNote(''); setError('');
    try {
      const body = new FormData(); body.append('file', file);
      const response = await fetch(`${API}/api/sources/${encodeURIComponent(filename)}/upload`, { method: 'POST', body });
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
      const join = result.join_diagnostic;
      const joinNote = join ? ` Student-ID match: ${join.match_pct}% (${nfmt(join.matched_student_ids)} matched, ${nfmt(join.unmatched_student_ids)} unmatched).${join.warning ? ` ${join.warning}` : ''}` : '';
      setSourceNote(`${filename}: ${result.message} Active analytics have not changed yet.${joinNote}`);
      setSources((await getJson('/api/sources')).items);
    } catch (e) { setError(e.message); } finally { setSyncingSource(''); }
  };
  const renderSources = () => <>
    <div className="page-intro"><div><div className="eyebrow">ATLAS / CAMPUS DATA INTEGRATION</div><h1>Connect the <span>campus.</span></h1><p>Manage campus data files, validate incoming records and refresh the unified student view without manually combining datasets.</p></div><button className="btn btn-dark" onClick={async () => { setSyncingSource('*'); try { const result = await getJson('/api/sources/sync-all', { method: 'POST' }); setSources((await getJson('/api/sources')).items); setQuality(null); await loadCore(); setSourceNote(`${result.message} ${nfmt(result.student_count)} student profiles refreshed. ${analyticsDeltaText(result)}`); } catch(e) { setError(e.message); } finally { setSyncingSource(''); } }} disabled={!!syncingSource}><RefreshCw size={15} className={syncingSource ? 'spin' : ''} /> Sync all sources</button></div>
    <div className="source-banner"><div className="source-banner-icon"><Database size={22}/></div><div><b>Integration-first architecture</b><p>Upload a CSV for any listed source, review validation, then choose Activate & refresh. ATLAS recalculates charts, scores, segments and insights from the accepted source together with the other active files.</p></div><span className="demo-tag">DEMO MODE</span></div>
    {sourceNote && <div className="success-banner"><CheckCircle2 size={17}/><span>{sourceNote}</span><button onClick={() => setSourceNote('')}><X size={14}/></button></div>}
    <div className="source-grid">{sources.map((src, i) => <section className="panel source-card" key={src.filename}><div className="source-card-top"><div className="source-file"><Database size={17}/></div><span className={`status-pill ${src.staged_upload ? 'staged' : src.status.toLowerCase()}`}><i/>{src.staged_upload ? 'Upload ready' : src.status}</span></div><div className="eyebrow">{src.group}</div><h3>{src.label}</h3><p>{src.filename}</p><div className="source-stats"><div><span>{src.staged_upload ? 'Uploaded rows' : 'Active rows'}</span><b>{nfmt(src.staged_upload ? src.staged_row_count : src.row_count)}</b></div><div><span>{src.staged_upload ? 'Active rows' : 'Connector'}</span><b>{src.staged_upload ? nfmt(src.row_count) : src.connection_mode}</b></div></div><div className="source-sync-meta"><Clock3 size={13}/>{src.last_sync ? `Last sync ${new Date(src.last_sync).toLocaleString()}` : 'Bundled demo source · not yet synced'}</div><div className="source-card-actions"><button className="btn btn-dark" onClick={() => syncSource(src.filename)} disabled={!!syncingSource}><RefreshCw size={14} className={syncingSource === src.filename ? 'spin' : ''}/> {src.staged_upload ? 'Activate & refresh' : 'Refresh analytics'}</button><label className="btn btn-secondary upload-label"><Upload size={14}/> Validate CSV<input type="file" accept=".csv,text/csv" onChange={e => { uploadSource(src.filename, e.target.files?.[0]); e.target.value = ''; }} hidden /></label></div><small className="source-hint">{src.staged_upload ? 'A validated upload is waiting. Activate it to replace the current source and recalculate analytics.' : 'Upload a CSV to replace this source, or refresh analytics using the active file.'}</small></section>)}</div>
    <section className="panel connector-roadmap"><div className="card-head"><div><h3>Connector framework</h3><p>Production-ready direction without pretending external systems are already connected.</p></div></div><div className="connector-types"><div><Cable size={19}/><b>Connected system</b><span>Secure access · reliable data updates</span><em>Ready to configure · not connected</em></div><div><Database size={19}/><b>Database connection</b><span>Institution-approved credentials · scoped views</span><em>Ready to configure · not connected</em></div><div><Clock3 size={19}/><b>Scheduled data import</b><span>Column checks · sync history · retries</span><em>Local file validation implemented</em></div><div><Activity size={19}/><b>Automatic data updates</b><span>New-record notifications · duplicate-safe updates</span><em>Future connector</em></div></div></section>
  </>;

  const renderQuality = () => <>
    <div className="page-intro"><div><div className="eyebrow">CAMPUS INTELLIGENCE / DATA QUALITY</div><h1>Trust the <span>inputs.</span></h1><p>Inspect how the disconnected source files are validated and connected before results reach decision-makers.</p></div><button className="btn btn-secondary" onClick={() => getJson('/api/data-quality').then(setQuality).catch(e => setError(e.message))}><RefreshCw size={15} /> Run checks</button></div>
    {!quality ? <Loading /> : <><div className="quality-summary"><div className="quality-big"><span className="quality-mark"><Database size={23} /></span><div><div className="eyebrow">UNIFIED STUDENT VIEW</div><h2>{nfmt(quality.student_count)} profiles</h2><p>{quality.sources.length} source tables inspected · joined via Student_ID</p></div></div><div className="quality-mini"><span>Profiles with feedback</span><b>{nfmt(quality.students_with_feedback)}</b><small>{pct(100 * quality.students_with_feedback / Math.max(1, quality.student_count))} response coverage</small></div><div className="quality-mini"><span>Master ID duplicates</span><b>{nfmt(quality.duplicate_master_ids)}</b><small>{quality.duplicate_master_ids === 0 ? 'Unique-key check passed' : 'Needs review'}</small></div></div><section className="panel"><CardHeader title="Source integration monitor" subtitle="Separate source tables are aggregated to student level before joining. This prevents one-to-many row multiplication." /><div className="table-scroll"><table className="data-table quality-table"><thead><tr><th>Source table</th><th>Rows</th><th>Columns</th><th>Duplicate key rows</th><th>Orphan student rows</th><th>Status</th></tr></thead><tbody>{quality.sources.map(src => <tr key={src.source}><td><div className="source-name"><span className="source-file"><Database size={15} /></span><div><b>{src.source}</b><small>{src.source === 'student_master.csv' ? 'Primary student table' : src.source === 'course_catalog.csv' ? 'Course reference' : 'Supplementary source'}</small></div></div></td><td>{nfmt(src.row_count)}</td><td>{src.column_count}</td><td>{nfmt(src.duplicate_key_rows)}</td><td>{nfmt(src.orphan_student_rows)}</td><td><span className={`status-pill ${src.status.toLowerCase()}`}><span />{src.status}</span></td></tr>)}</tbody></table></div></section><section className="panel join-panel"><CardHeader title="Source coverage" subtitle="Coverage is calculated against the current student master set. Feedback can legitimately have lower coverage when students did not respond." /><div className="coverage-grid">{quality.join_coverage.map(j => <div className="coverage-item" key={j.source}><div className="coverage-title"><b>{j.source}</b><span>{pct(j.coverage_pct_of_master)}</span></div><div className="coverage-bar"><i style={{ width: `${j.coverage_pct_of_master}%` }} /></div><div className="coverage-meta"><span>{nfmt(j.matched_student_ids)} matched student IDs</span><span>{nfmt(j.orphan_student_ids)} orphans</span></div></div>)}</div></section><div className="quality-notes">{quality.notes.map((note, i) => <div key={note}><CircleCheck size={17} /><span>{note}</span></div>)}</div><div className="provenance-note"><AlertTriangle size={17} /><div><b>Provenance is part of the product.</b><span>The entire bundle is synthetic demo data. Use it to demonstrate integration, analytics and UX; do not treat these results as proven for real students.</span></div></div></>}
  </>;

  const renderAskAtlas = () => <>
    <div className="page-intro"><div><div className="eyebrow">ATLAS / CAMPUS QUESTIONS</div><h1>Ask for the <span>evidence.</span></h1><p>Ask a plain-language question about the active campus data. ATLAS calculates the facts; Gemini, when configured, only explains those facts.</p></div></div>
    <section className="panel" style={{padding:'24px',maxWidth:'980px'}}>
      <label style={{display:'block',fontWeight:700,marginBottom:'10px'}}>What would you like to understand?</label>
      <textarea value={atlasQuestion} onChange={e=>setAtlasQuestion(e.target.value)} rows={3} maxLength={1000} style={{width:'100%',padding:'14px',border:'1px solid #d9e1e4',borderRadius:'12px',font:'inherit',resize:'vertical'}} placeholder="e.g. Which student groups show a placement-readiness gap?" />
      <div style={{display:'flex',gap:'10px',alignItems:'center',marginTop:'12px',flexWrap:'wrap'}}><button className="btn btn-dark" disabled={atlasBusy || atlasQuestion.trim().length<3} onClick={async()=>{setAtlasBusy(true);setAtlasAnswer(null);try{setAtlasAnswer(await getJson('/api/ai/ask',{method:'POST',body:JSON.stringify({question:atlasQuestion})}));}catch(e){setAtlasAnswer({answer:`ATLAS could not answer this request: ${e.message}`,mode:'request failed'});}finally{setAtlasBusy(false);}}}>{atlasBusy?'Reviewing current data…':'Ask ATLAS'} <ArrowUpRight size={15}/></button><span className="footnote">No API key? ATLAS still returns calculated findings. Configure Gemini on the backend for AI-written summaries.</span></div>
    </section>
    {atlasAnswer && <section className="panel" style={{padding:'24px',maxWidth:'980px',marginTop:'18px'}}><div className="eyebrow">{atlasAnswer.mode || 'ATLAS RESPONSE'}</div><h2 style={{marginTop:'8px'}}>What the current data says</h2><p style={{whiteSpace:'pre-wrap',lineHeight:1.7}}>{atlasAnswer.answer}</p>{atlasAnswer.note && <p className="footnote">{atlasAnswer.note}</p>}{atlasAnswer.patterns?.length>0 && <><h3>Related findings</h3><div className="pattern-grid">{atlasAnswer.patterns.slice(0,4).map(item=><article className="panel pattern-card" key={item.id}><div className="eyebrow">{item.category}</div><h3>{item.title}</h3><p>{item.interpretation}</p><p><b>Evidence:</b> {JSON.stringify(item.evidence)}</p><p><b>Suggested review:</b> {item.suggested_action}</p></article>)}</div></>}</section>}
  </>;

  const renderReports = () => <>
    <div className="page-intro"><div><div className="eyebrow">CAMPUS INTELLIGENCE / DECISION SUPPORT</div><h1>From signals <span>to next steps.</span></h1><p>Generate an evidence-backed institutional brief from the validated analytics layer.</p></div></div>
    <div className="report-layout"><section className="panel report-builder"><div className="report-title-icon"><FileBarChart2 size={22} /></div><div className="eyebrow">REPORT BUILDER</div><h2>Ask better questions of campus data.</h2><p>Choose the scope. ATLAS calculates each figure from the current records and assembles a report with clear limitations.</p><label className="field-label">Report type<select value={reportType} onChange={e => setReportType(e.target.value)}><option value="risk_snapshot">Student success risk snapshot</option><option value="institutional_brief">Institutional decision brief</option></select></label><label className="field-label">Department / branch<select value={reportBranch} onChange={e => setReportBranch(e.target.value)}>{branches.map(b => <option key={b}>{b}</option>)}</select></label><button className="btn btn-lime full-btn" onClick={generateReport} disabled={busy}>{busy ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />} Generate report <ArrowUpRight size={15} /></button><div className="builder-foot"><ShieldAlert size={14} /> Figures are calculated from the active dataset; each finding includes context and limitations.</div></section><section className="report-output">{!report ? <div className="report-empty"><div className="report-empty-icon"><FileBarChart2 size={28} /></div><h3>Your decision brief appears here.</h3><p>Generate a report to see campus metrics, notable patterns, recommended checks and data limitations.</p><div className="report-example"><span>EXAMPLE PROMPT</span><b>“Summarize placement readiness gaps for CSE and what the department should investigate first.”</b></div></div> : <div className="panel report-result"><div className="report-result-head"><div><div className="eyebrow">GENERATED FROM VERIFIED METRICS</div><h2>{report.title}</h2><p>{report.summary}</p></div><button className="icon-button" onClick={() => navigator.clipboard?.writeText(JSON.stringify(report, null, 2))} title="Copy report data"><ClipboardList size={17} /></button></div><div className="report-scope"><span>{report.scope.branch === 'All' ? 'All branches' : report.scope.branch}</span><span>{nfmt(report.scope.students)} student profiles</span><span>synthetic dataset</span></div><div className="report-means">{Object.entries(report.mean_scores || {}).map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><b>{one(value)}</b></div>)}</div><h3 className="report-section-title">Findings to review</h3><div className="finding-list">{report.findings.map((f, i) => <div key={f.title} className="finding"><div className="finding-number">0{i + 1}</div><div className="finding-main"><b>{f.title}</b><p>{f.detail}</p></div><div className="finding-metric"><b>{nfmt(f.metric)}</b><span>{f.unit}</span></div></div>)}</div><h3 className="report-section-title">Recommended checks</h3><div className="recommendation-list">{report.recommended_checks.map(x => <div key={x}><Check size={15} />{x}</div>)}</div><div className="report-limits"><AlertTriangle size={16} /><div><b>Limitations</b>{report.limitations.map(x => <span key={x}>{x}</span>)}</div></div></div>}</section></div>
  </>;

  const title = NAV.find(n => n.id === page)?.label || 'Overview';
  useEffect(() => { if (!allowedPages.includes(page)) setPage(defaultPage); }, [workspace?.role]);
  return <div className="app-shell">
    <aside className="sidebar"><div className="brand"><div className="brand-mark"><span>A</span><i /></div><div><b>ATLAS<span> /</span></b><small>STUDENT SUCCESS INTELLIGENCE</small></div></div><div className="workspace-chip"><div className="workspace-logo">K</div><div><b>{workspace?.institution || 'ATLAS Demo Campus'}</b><small>{workspace?.role || 'Institution admin'}</small></div><ChevronDown size={15} /></div><div className="nav-label">WORKSPACE</div><nav className="side-nav">{visibleNav.map(item => <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => setPage(item.id)}><item.icon size={18} strokeWidth={1.8} /><span>{item.label}</span>{item.id === 'quality' && <span className="nav-indicator" />}</button>)}</nav><div className="sidebar-bottom"><div className="sidebar-tip"><div className="tip-icon"><Lightbulb size={16} /></div><b>Evidence before action.</b><p>Risk flags help decide what to review. They are not labels for students.</p></div><div className="profile-mini"><div className="profile-avatar">S</div><div><b>Success team</b><small>Prototype workspace</small></div><button title="Help"><CircleHelp size={17} /></button></div></div></aside>
    <main className="main"><header className="topbar"><div className="breadcrumb"><span>Workspace</span><ChevronRight size={13} /><b>{title}</b></div><div className="topbar-actions"><span className="connection"><i /> Workspace data connected</span><button className="icon-button" title="Notifications" onClick={() => setError('There are no live alerts in this synthetic-data prototype.')}><Bell size={18} /></button><div className="topbar-avatar">S</div></div></header><div className="content">{error && <div className="error-banner"><AlertTriangle size={17} /><span>{error}</span><button onClick={() => setError('')}><X size={15} /></button></div>}{page === 'overview' && renderOverview()}{page === 'students' && renderStudents()}{page === 'riskpatterns' && renderRiskPatterns()}{page === 'explanations' && renderExplanations()}{page === 'actions' && renderActions()}{page === 'interventions' && renderInterventions()}{page === 'segments' && renderSegments()}{page === 'sources' && renderSources()}{page === 'quality' && renderQuality()}{page === 'reports' && renderReports()}{page === 'askatlas' && renderAskAtlas()}<footer className="app-footer"><span>STUDENT SUCCESS INTELLIGENCE <i>·</i> PROTOTYPE 0.1</span><span>{lastUpdated ? `Data loaded ${lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : 'Connecting to API…'} <i>·</i> <span className="footer-dot" /> Synthetic data</span></footer></div></main>
  </div>;
}

function StudentProfile({ detail }) {
  const [modelExplanation, setModelExplanation] = useState(null);
  useEffect(() => { let active = true; setModelExplanation(null); const load = async () => { try { let r = await fetch(`${API}/api/students/${encodeURIComponent(detail.student_id)}/shap-explanation`); let data = r.ok ? await r.json() : null; if (!data) { r = await fetch(`${API}/api/students/${encodeURIComponent(detail.student_id)}/placement-explanation`); data = r.ok ? await r.json() : null; } if (active) setModelExplanation(data); } catch { if (active) setModelExplanation(null); } }; load(); return () => { active = false; }; }, [detail.student_id]);
  return <><div className="profile-drawer-head"><div className="profile-drawer-avatar">{detail.student_id.slice(-2)}</div><div><div className="eyebrow">STUDENT INTELLIGENCE PROFILE</div><h2>{detail.student_id}</h2><p>{detail.branch} · Program year {detail.program_year ?? '—'}</p></div></div><div className="profile-score-hero"><div><span>Student Success Score</span><div className="large-score">{one(detail.success_score)}<small>/100</small></div><span className="segment-text">{detail.segment}</span></div><ScoreRing value={detail.success_score} size={86} /></div><div className="profile-risks"><div><span>Academic risk</span><RiskPill level={detail.academic_risk} /></div><div><span>Placement risk</span><RiskPill level={detail.placement_risk} /></div><div><span>Estimated placement risk</span><b>{detail.placement_risk_probability == null ? 'Unavailable' : pct(detail.placement_risk_probability)}</b></div></div><div className="profile-component-scores">{[['Academic', detail.academic_score], ['Engagement', detail.engagement_score], ['Placement readiness', detail.placement_readiness_score]].map(([label, value]) => <div key={label}><div><span>{label}</span><b>{one(value)}</b></div><div className="mini-progress"><i style={{ width: `${value}%` }} /></div></div>)}</div><div className="profile-section"><h3>Signals to review</h3>{detail.explanation_drivers.length ? <div className="driver-list">{detail.explanation_drivers.map(d => <div key={d}><span><Activity size={14} /></span>{d}</div>)}</div> : <div className="no-drivers"><CircleCheck size={16} /> No configured rule-based indicators triggered. Continue normal monitoring.</div>}<p className="caveat-text">These drivers describe recorded signals. They do not prove causation or guarantee an outcome.</p></div><div className="profile-section"><h3>Learning activity · 16-week view</h3><div className="profile-mini-chart"><ResponsiveContainer width="100%" height={150}><AreaChart data={detail.weekly_lms_activity} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}><CartesianGrid vertical={false} stroke="#eee" /><XAxis dataKey="week" tick={{ fontSize: 9 }} axisLine={false} tickLine={false} /><YAxis domain={[0, 100]} tick={{ fontSize: 9 }} axisLine={false} tickLine={false} /><Tooltip /><Area dataKey="activity_index" name="Activity" type="monotone" stroke="#22251b" fill="#d6fb45" fillOpacity={0.55} /></AreaChart></ResponsiveContainer></div></div><div className="profile-section"><h3>Course records</h3><div className="course-list">{detail.course_records.map(c => <div key={c.course_id}><span><b>{c.course_name}</b><small>{c.course_id} · Semester {c.semester}</small></span><b className={c.backlog ? 'bad-text' : ''}>{one(c.marks)}{c.backlog && <small> · BACKLOG</small>}</b></div>)}</div></div><div className="profile-section"><h3>Placement estimate · evidence</h3>{modelExplanation ? <><p className="caveat-text">Estimated non-placement: <b>{pct(modelExplanation.estimated_non_placement_pct)}</b>. Explanation method: {modelExplanation.method || 'model feature contributions'}. Attributions describe model behaviour, not causes.</p><div className="driver-list">{modelExplanation.top_contributions.slice(0,6).map((item,i)=>{const value = item.shap_value ?? item.contribution ?? 0; return <div key={`${item.feature}-${i}`}><span><Activity size={14}/></span><div><b>{item.feature}</b><small>{item.direction} · attribution {value > 0 ? '+' : ''}{value}</small></div></div>})}</div><p className="caveat-text">{modelExplanation.caveat}</p></> : <p className="caveat-text">Model explanation unavailable for this profile. Rule-based signals above remain available.</p>}</div><div className="profile-section"><h3>Scoring method</h3><div className="method-note">{detail.score_method.success}<br /><br />{detail.score_method.caveat}</div></div></>;
}


function PublicSite({ onSignIn, onDemo }) {
  return <div className="public-site">
    <header className="marketing-nav"><a className="marketing-brand" href="#top"><span className="marketing-mark">A<i /></span><span>ATLAS<small>STUDENT SUCCESS INTELLIGENCE</small></span></a><nav><a href="#platform">Platform</a><a href="#how-it-works">How it works</a><a href="#for-institutions">For institutions</a></nav><div className="marketing-actions"><button className="nav-login" onClick={onSignIn}>Sign in</button><button className="btn btn-dark marketing-cta" onClick={onSignIn}>Get started <ArrowUpRight size={16}/></button></div></header>
    <main id="top">
      <section className="marketing-hero"><div className="hero-copy"><div className="marketing-kicker"><span/> A clearer view of student success</div><h1>See the signal.<br/><em>Shape the future.</em></h1><p>Bring campus data into focus. Understand where students need support, uncover patterns across courses, and help teams take the next right step.</p><div className="hero-buttons"><button className="btn btn-dark hero-primary" onClick={onSignIn}>Explore ATLAS <ArrowUpRight size={17}/></button><button className="hero-secondary" onClick={onDemo}><span className="play-chip">▶</span> Explore the demo</button></div><div className="hero-proof"><div className="proof-avatars"><span>F</span><span>A</span><span>P</span></div><span>Built for the people who shape student outcomes</span></div></div>
        <div className="hero-visual"><div className="visual-orbit orbit-one"/><div className="visual-orbit orbit-two"/><div className="float-sticker sticker-top">LESS GUESSWORK <span>↗</span></div><div className="dashboard-mock"><div className="mock-top"><div className="mock-brand"><span>A</span> ATLAS</div><div className="mock-dots"><i/><i/><i/></div></div><div className="mock-welcome"><div><small>INSTITUTION OVERVIEW</small><h3>Student success,<br/>in perspective.</h3></div><span className="mock-period">This semester⌄</span></div><div className="mock-stats"><div><small>Students in view</small><b>12,480</b><span>Across 8 departments</span></div><div className="mock-stat-lime"><small>Need a closer look</small><b>08.6%</b><span>Prioritised for review ↗</span></div></div><div className="mock-chart-card"><div className="mock-chart-head"><span>Student momentum</span><small>Last 6 weeks ↗</small></div><div className="mock-chart"><div className="chart-gridline g1"/><div className="chart-gridline g2"/><div className="chart-gridline g3"/><svg viewBox="0 0 440 110" preserveAspectRatio="none" aria-label="Illustrative trend chart"><path d="M0,82 C28,76 40,60 65,67 S106,82 132,56 S178,68 205,42 S248,57 276,34 S320,50 345,24 S400,33 440,12" fill="none" stroke="#1b1d18" strokeWidth="3.5" strokeLinecap="round"/><path d="M0,98 C28,89 40,83 65,85 S106,91 132,78 S178,80 205,68 S248,77 276,65 S320,69 345,57 S400,60 440,48" fill="none" stroke="#b6d92e" strokeWidth="3" strokeLinecap="round" strokeDasharray="4 5"/></svg></div><div className="mock-chart-labels"><span>Wk 01</span><span>Wk 02</span><span>Wk 03</span><span>Wk 04</span><span>Wk 05</span><span>Wk 06</span></div></div><div className="mock-insight"><span className="insight-icon"><Sparkles size={17}/></span><div><small>A PATTERN TO EXPLORE</small><b>Course engagement is dipping</b><span>Review the affected cohorts before planning support.</span></div><ArrowUpRight size={17}/></div></div><div className="float-sticker sticker-bottom"><span className="sticker-spark">✳</span><div><b>Insight → Action</b><small>Keep the next step in view</small></div></div><div className="hero-shape shape-dot">✳</div><div className="hero-shape shape-square"/></div>
      </section>
      <section className="trust-strip"><span>ONE CLEARER PICTURE</span><div><i>↗</i> Academics</div><div><i>◌</i> Attendance</div><div><i>▤</i> Learning activity</div><div><i>◎</i> Placement readiness</div><div><i>✳</i> Student voice</div></section>
      <section className="marketing-section" id="platform"><div className="section-intro"><div className="marketing-kicker"><span/> THE ATLAS APPROACH</div><h2>Not another dashboard.<br/><em>A way forward.</em></h2><p>Data becomes useful when it helps people understand what matters and decide what to do next.</p></div><div className="feature-grid"><article className="feature-card feature-dark"><div className="feature-top"><span>01 / CONNECT</span><span className="feature-symbol">↗</span></div><div className="feature-graphic connect-graphic"><div className="node node-a">ACADEMICS</div><div className="node node-b">ATTENDANCE</div><div className="node node-c">LEARNING</div><div className="node node-d">PLACEMENT</div><div className="node-center">A</div><div className="connect-line line-a"/><div className="connect-line line-b"/><div className="connect-line line-c"/><div className="connect-line line-d"/></div><h3>One student.<br/>A fuller picture.</h3><p>Bring disconnected campus records together into a view teams can actually use.</p></article><article className="feature-card feature-lime"><div className="feature-top"><span>02 / UNDERSTAND</span><span className="feature-symbol">↗</span></div><div className="feature-graphic score-graphic"><div className="score-orb"><b>76</b><small>IN FOCUS</small></div><div className="score-tag tag-one">Academic signals <i>↗</i></div><div className="score-tag tag-two">Readiness gap <i>↗</i></div><div className="score-tag tag-three">Engagement <i>↗</i></div></div><h3>Look beyond<br/>the average.</h3><p>See the signals behind a score, discover meaningful groups, and surface patterns worth investigating.</p></article><article className="feature-card feature-white"><div className="feature-top"><span>03 / RESPOND</span><span className="feature-symbol">↗</span></div><div className="feature-graphic action-graphic"><div className="action-mini"><span className="action-check">✓</span><div><b>Review attendance trend</b><small>Faculty mentor · Assigned</small></div><span className="action-arrow">↗</span></div><div className="action-mini"><span className="action-check">✓</span><div><b>Plan a practice cohort</b><small>Placement team · In progress</small></div><span className="action-arrow">↗</span></div><div className="action-mini action-muted"><span className="action-check">○</span><div><b>Check progress again</b><small>Follow-up · Upcoming</small></div></div></div><h3>Insight that<br/>moves people.</h3><p>Help teams review, assign and follow up on support—without turning a risk flag into a verdict.</p></article></div></section>
      <section className="statement-band" id="how-it-works"><div className="statement-symbol">✳</div><div><small>THE IDEA IN ONE LINE</small><h2>From scattered records<br/>to <em>considered action.</em></h2></div><button className="round-arrow" onClick={onSignIn}><ArrowUpRight size={23}/></button><div className="statement-decoration">A<span>.</span></div></section>
      <section className="workflow-section" id="for-institutions"><div className="section-intro"><div className="marketing-kicker"><span/> MADE FOR CAMPUS TEAMS</div><h2>Different roles.<br/><em>One shared picture.</em></h2><p>Give every team the view they need, scoped to their responsibilities.</p></div><div className="role-grid"><div><span className="role-icon">⌂</span><b>Institution leaders</b><p>See campus-wide trends and decide where attention is needed most.</p></div><div><span className="role-icon">◫</span><b>Department heads</b><p>Explore cohort patterns and course-level bottlenecks.</p></div><div><span className="role-icon">✳</span><b>Faculty & mentors</b><p>Review assigned students, understand signals and follow up.</p></div><div><span className="role-icon">↗</span><b>Placement teams</b><p>Spot readiness gaps and plan focused preparation.</p></div></div></section>
      <section className="final-cta"><div className="cta-shape">✳</div><div className="marketing-kicker"><span/> THE NEXT STEP STARTS HERE</div><h2>Make student data<br/><em>mean something.</em></h2><p>Bring the pieces together. Find what deserves attention. Help your campus move forward.</p><button className="btn btn-dark hero-primary" onClick={onSignIn}>Enter your workspace <ArrowUpRight size={17}/></button><small>Designed for institution teams · Demo data available</small></section>
    </main><footer className="marketing-footer"><a className="marketing-brand" href="#top"><span className="marketing-mark">A<i/></span><span>ATLAS<small>STUDENT SUCCESS INTELLIGENCE</small></span></a><span>From student data to institutional action.</span><span>© ATLAS · Prototype</span></footer>
  </div>;
}

function AuthScreen({ onBack, onEnter }) {
  const [mode, setMode] = useState('signin');
  const [institution, setInstitution] = useState('ATLAS Demo Campus');
  const [email, setEmail] = useState('admin@atlas.demo');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('Institution admin');
  const [error, setError] = useState('');
  const demoAccounts = {
    'admin@atlas.demo': { role: 'Institution admin', password: 'AtlasDemo@2026' },
    'hod@atlas.demo': { role: 'Department head', password: 'AtlasDemo@2026' },
    'faculty@atlas.demo': { role: 'Faculty / mentor', password: 'AtlasDemo@2026' },
    'placement@atlas.demo': { role: 'Placement officer', password: 'AtlasDemo@2026' },
  };
  const submit = (e) => {
    e.preventDefault(); setError('');
    if (!institution.trim() || !email.includes('@')) { setError('Enter a valid institution and email address.'); return; }
    if (mode === 'signin') {
      const account = demoAccounts[email.trim().toLowerCase()];
      if (!account || account.password !== password) { setError('Use one of the four demo accounts shown below with its demo password.'); return; }
      onEnter({ institution, email: email.trim().toLowerCase(), role: account.role, mode: 'demo' });
      return;
    }
    if (password.length < 4) { setError('Enter a password with at least 4 characters.'); return; }
    onEnter({ institution, email, role, mode });
  };
  return <div className="auth-screen"><button className="auth-back" onClick={onBack}>← Back to ATLAS</button><div className="auth-brand"><span className="marketing-mark">A<i/></span><span>ATLAS<small>STUDENT SUCCESS INTELLIGENCE</small></span></div><div className="auth-layout"><div className="auth-story"><div className="marketing-kicker"><span/> YOUR CAMPUS, IN FOCUS</div><h1>A clearer view<br/>starts <em>here.</em></h1><p>Sign in to explore your institution's student success workspace, or create a workspace to get started with your campus data.</p><div className="auth-benefits"><div><span>01</span><b>Your data, in one view</b></div><div><span>02</span><b>Patterns worth a closer look</b></div><div><span>03</span><b>Actions you can follow through</b></div></div><div className="auth-note"><ShieldAlert size={16}/> Prototype note: authentication is a demo flow.</div></div><form className="auth-card" onSubmit={submit}><div className="auth-card-icon"><span>✳</span></div><h2>{mode === 'signin' ? 'Welcome back.' : 'Create your workspace.'}</h2><p>{mode === 'signin' ? 'Sign in to continue to your campus view.' : 'Set up a workspace for your institution.'}</p><div className="auth-tabs"><button type="button" className={mode === 'signin' ? 'selected' : ''} onClick={() => setMode('signin')}>Sign in</button><button type="button" className={mode === 'signup' ? 'selected' : ''} onClick={() => setMode('signup')}>New workspace</button></div><label>Institution name<input value={institution} onChange={e => setInstitution(e.target.value)} placeholder="Your college or university" required/></label><label>Work email<input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@institution.edu" required/></label><label>Password<input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Enter password" required minLength={4}/></label><label>{mode === 'signup' ? 'Your role' : 'Preview workspace as'}<select value={role} onChange={e => setRole(e.target.value)}><option>Institution admin</option><option>Department head</option><option>Faculty / mentor</option><option>Placement officer</option></select></label>{error && <div className="auth-error">{error}</div>}<button className="btn btn-dark auth-submit" type="submit">{mode === 'signin' ? 'Continue to workspace' : 'Create workspace'} <ArrowUpRight size={17}/></button><div className="demo-credentials"><b>Demo accounts · password for all: <code>AtlasDemo@2026</code></b><span>Admin: <code>admin@atlas.demo</code></span><span>Department Head: <code>hod@atlas.demo</code></span><span>Faculty / Mentor: <code>faculty@atlas.demo</code></span><span>Placement Officer: <code>placement@atlas.demo</code></span></div><small className="auth-footnote">Demo-only sign-in. Role views are for evaluation.</small></form></div></div>;
}

export default function App() {
  const [screen, setScreen] = useState('landing');
  const [workspace, setWorkspace] = useState(null);
  const enter = (user) => { setWorkspace(user); setScreen('dashboard'); };
  if (screen === 'landing') return <PublicSite onSignIn={() => setScreen('auth')} onDemo={() => enter({ institution: 'ATLAS Demo Campus', email: 'demo@atlas.local', role: 'Institution admin', mode: 'demo' })}/>;
  if (screen === 'auth') return <AuthScreen onBack={() => setScreen('landing')} onEnter={enter}/>;
  return <div className="workspace-app"><div className="workspace-banner"><span className="workspace-live-dot"/> <b>{workspace?.institution || 'ATLAS Demo Campus'}</b><span>·</span><span>{workspace?.role || 'Institution admin'}</span><button onClick={() => setScreen('landing')}>Exit workspace ↗</button></div><Dashboard workspace={workspace}/></div>;
}
