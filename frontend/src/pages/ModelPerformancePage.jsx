import React, { useState, useEffect } from 'react';
import {
  Activity,
  BrainCircuit,
  TrendingUp,
  CheckCircle2,
  AlertTriangle,
  RotateCw,
  Info,
  ShieldAlert,
  GitBranch,
  Layers,
  HelpCircle
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';
import { modelService } from '../services/api';

export default function ModelPerformancePage() {
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState(null);

  useEffect(() => {
    fetchComparison();
  }, []);

  const fetchComparison = async () => {
    setLoading(true);
    try {
      const data = await modelService.getComparison();
      setComparison(data);
    } catch (err) {
      console.error('Failed to load model comparison', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    setRetraining(true);
    setRetrainMsg(null);
    try {
      const res = await modelService.retrain();
      setRetrainMsg({ type: 'success', message: res.message });
      await fetchComparison();
    } catch (err) {
      setRetrainMsg({ type: 'error', message: 'Retraining failed. Check backend logs.' });
    } finally {
      setRetraining(false);
    }
  };

  if (loading && !comparison) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <div className="w-12 h-12 border-4 border-sky-600 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-slate-500 font-medium">Extracting Model Evaluation Metrics & Curves...</p>
      </div>
    );
  }

  const { models, top_features, imbalance_explanation, selected_best_model } = comparison || {};
  const baseline = models?.['Naive Majority Baseline'];
  const dt = models?.['Decision Tree Classifier'];
  const rf = models?.['Random Forest Classifier'];

  // Combine ROC curves for charts
  const rocChartData = (rf?.metrics?.roc_curve || []).map((pt, i) => {
    const dtPt = dt?.metrics?.roc_curve?.[i] || {};
    return {
      fpr: pt.fpr,
      rf_tpr: pt.tpr,
      dt_tpr: dtPt.tpr || pt.fpr,
      baseline_tpr: pt.fpr, // Random chance diagonal line
    };
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Title & Retrain Header */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <Activity className="w-5 h-5 text-sky-600" />
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              Machine Learning Model Evaluation & Comparison
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Genuine test set evaluation (294 records) comparing Naive Baseline, Decision Tree, and Random Forest.
          </p>
        </div>

        <button
          onClick={handleRetrain}
          disabled={retraining}
          className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-semibold flex items-center space-x-2 transition shadow-sm disabled:opacity-50"
        >
          <RotateCw className={`w-3.5 h-3.5 ${retraining ? 'animate-spin' : ''}`} />
          <span>{retraining ? 'Retraining Models...' : 'Retrain Live Models'}</span>
        </button>
      </div>

      {retrainMsg && (
        <div
          className={`p-4 rounded-xl border text-xs flex items-center space-x-2 ${
            retrainMsg.type === 'success'
              ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
              : 'bg-rose-50 border-rose-200 text-rose-800'
          }`}
        >
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{retrainMsg.message}</span>
        </div>
      )}

      {/* Class Imbalance Paradox Callout */}
      <div className="bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-transparent border border-amber-300/80 rounded-2xl p-5 text-amber-950">
        <div className="flex items-start space-x-3">
          <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <h3 className="font-bold text-sm text-amber-900">
              The Imbalanced Classification Paradox (DMBI Concept)
            </h3>
            <p className="text-xs text-amber-800 mt-1 leading-relaxed">{imbalance_explanation}</p>
          </div>
        </div>
      </div>

      {/* Side-by-Side Model Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Model 1: Naive Majority Baseline */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Baseline Model
              </span>
              <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 text-[10px] font-semibold">
                Majority Class
              </span>
            </div>
            <h4 className="text-base font-bold text-slate-900">Naive Majority Baseline</h4>
            <p className="text-xs text-slate-500 mt-1 mb-4 leading-relaxed">{baseline?.description}</p>

            <div className="space-y-2.5 border-t border-slate-100 pt-3">
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Accuracy:</span>
                <span className="font-bold text-amber-600 font-mono">
                  {(baseline?.metrics?.accuracy * 100).toFixed(1)}% (Deceptive)
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Precision:</span>
                <span className="font-bold text-rose-600 font-mono">0.0%</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Recall:</span>
                <span className="font-bold text-rose-600 font-mono">0.0% (Zero Leavers Caught)</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">F1-Score:</span>
                <span className="font-bold text-rose-600 font-mono">0.0000</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">ROC-AUC:</span>
                <span className="font-bold text-slate-700 font-mono">0.5000 (Random)</span>
              </div>
            </div>
          </div>

          {/* Baseline Confusion Matrix */}
          <div className="mt-5 pt-3 border-t border-slate-100">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2">
              Confusion Matrix (Test Set):
            </span>
            <div className="grid grid-cols-2 gap-1.5 text-center text-xs">
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-400">True Neg (Retained)</div>
                <div className="font-bold text-slate-800">{baseline?.metrics?.confusion_matrix?.tn}</div>
              </div>
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-400">False Pos (False Alarm)</div>
                <div className="font-bold text-slate-800">{baseline?.metrics?.confusion_matrix?.fp}</div>
              </div>
              <div className="bg-rose-50 p-2 rounded-lg border border-rose-200">
                <div className="text-[10px] text-rose-500 font-semibold">False Neg (Missed Leavers)</div>
                <div className="font-bold text-rose-700">{baseline?.metrics?.confusion_matrix?.fn}</div>
              </div>
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-400">True Pos (Detected)</div>
                <div className="font-bold text-slate-800">{baseline?.metrics?.confusion_matrix?.tp}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Model 2: Decision Tree Classifier */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-bold uppercase tracking-wider text-sky-600">
                Algorithm A
              </span>
              <span className="px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200 text-[10px] font-semibold">
                Threshold: {dt?.threshold}
              </span>
            </div>
            <h4 className="text-base font-bold text-slate-900">Decision Tree Classifier</h4>
            <p className="text-xs text-slate-500 mt-1 mb-4 leading-relaxed">{dt?.description}</p>

            <div className="space-y-2.5 border-t border-slate-100 pt-3">
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Accuracy:</span>
                <span className="font-bold text-slate-800 font-mono">
                  {(dt?.metrics?.accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Precision:</span>
                <span className="font-bold text-sky-700 font-mono">
                  {(dt?.metrics?.precision * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Recall:</span>
                <span className="font-bold text-sky-700 font-mono">
                  {(dt?.metrics?.recall * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">F1-Score:</span>
                <span className="font-bold text-sky-700 font-mono">{dt?.metrics?.f1_score}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">ROC-AUC:</span>
                <span className="font-bold text-slate-800 font-mono">{dt?.metrics?.roc_auc}</span>
              </div>
            </div>
          </div>

          {/* Decision Tree Confusion Matrix */}
          <div className="mt-5 pt-3 border-t border-slate-100">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2">
              Confusion Matrix (Test Set):
            </span>
            <div className="grid grid-cols-2 gap-1.5 text-center text-xs">
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-400">TN (Retained)</div>
                <div className="font-bold text-slate-800">{dt?.metrics?.confusion_matrix?.tn}</div>
              </div>
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-400">FP (False Alarm)</div>
                <div className="font-bold text-slate-800">{dt?.metrics?.confusion_matrix?.fp}</div>
              </div>
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-400">FN (Missed)</div>
                <div className="font-bold text-slate-800">{dt?.metrics?.confusion_matrix?.fn}</div>
              </div>
              <div className="bg-emerald-50 p-2 rounded-lg border border-emerald-200">
                <div className="text-[10px] text-emerald-600 font-semibold">TP (Correct Leavers)</div>
                <div className="font-bold text-emerald-700">{dt?.metrics?.confusion_matrix?.tp}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Model 3: Random Forest Classifier (BEST) */}
        <div className="bg-white p-6 rounded-2xl border-2 border-teal-500 shadow-md flex flex-col justify-between relative overflow-hidden">
          <div className="absolute top-0 right-0 bg-teal-500 text-white text-[10px] font-bold px-3 py-1 rounded-bl-xl uppercase tracking-wider">
            Best Selected Model
          </div>
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-bold uppercase tracking-wider text-teal-600">
                Algorithm B (Ensemble)
              </span>
            </div>
            <h4 className="text-base font-bold text-slate-900">Random Forest Classifier</h4>
            <p className="text-xs text-slate-500 mt-1 mb-4 leading-relaxed">{rf?.description}</p>

            <div className="space-y-2.5 border-t border-slate-100 pt-3">
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Accuracy:</span>
                <span className="font-bold text-slate-800 font-mono">
                  {(rf?.metrics?.accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Precision:</span>
                <span className="font-bold text-teal-700 font-mono">
                  {(rf?.metrics?.precision * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Recall:</span>
                <span className="font-bold text-teal-700 font-mono">
                  {(rf?.metrics?.recall * 100).toFixed(1)}% (Superior)
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">F1-Score:</span>
                <span className="font-bold text-teal-700 font-mono font-bold">{rf?.metrics?.f1_score}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">ROC-AUC:</span>
                <span className="font-bold text-teal-700 font-mono">{rf?.metrics?.roc_auc}</span>
              </div>
            </div>
          </div>

          {/* Random Forest Confusion Matrix */}
          <div className="mt-5 pt-3 border-t border-slate-100">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2">
              Confusion Matrix (Test Set):
            </span>
            <div className="grid grid-cols-2 gap-1.5 text-center text-xs">
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-400">TN (Retained)</div>
                <div className="font-bold text-slate-800">{rf?.metrics?.confusion_matrix?.tn}</div>
              </div>
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-400">FP (False Alarm)</div>
                <div className="font-bold text-slate-800">{rf?.metrics?.confusion_matrix?.fp}</div>
              </div>
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-[10px] text-slate-400">FN (Missed)</div>
                <div className="font-bold text-slate-800">{rf?.metrics?.confusion_matrix?.fn}</div>
              </div>
              <div className="bg-teal-50 p-2 rounded-lg border border-teal-200">
                <div className="text-[10px] text-teal-600 font-semibold">TP (Correct Leavers)</div>
                <div className="font-bold text-teal-700">{rf?.metrics?.confusion_matrix?.tp}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Curves & Feature Importance Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROC Curves */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs">
          <div className="mb-4">
            <h3 className="font-bold text-slate-900 text-sm">Receiver Operating Characteristic (ROC) Curve</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Comparing Random Forest (ROC-AUC: {rf?.metrics?.roc_auc}) vs Decision Tree (ROC-AUC: {dt?.metrics?.roc_auc})
            </p>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rocChartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  dataKey="fpr"
                  label={{ value: 'False Positive Rate (FPR)', position: 'insideBottom', offset: -10, fontSize: 11 }}
                  tick={{ fill: '#64748B', fontSize: 11 }}
                />
                <YAxis
                  label={{ value: 'True Positive Rate (Recall)', angle: -90, position: 'insideLeft', fontSize: 11 }}
                  tick={{ fill: '#64748B', fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0F172A', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                <Line
                  type="monotone"
                  dataKey="rf_tpr"
                  name="Random Forest"
                  stroke="#06D6A0"
                  strokeWidth={2.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="dt_tpr"
                  name="Decision Tree"
                  stroke="#0284C7"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="baseline_tpr"
                  name="Random Baseline (0.5 AUC)"
                  stroke="#94A3B8"
                  strokeDasharray="4 4"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top 15 Feature Importances */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs">
          <div className="mb-4">
            <h3 className="font-bold text-slate-900 text-sm">Top 15 Predictive Attrition Drivers</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Derived from Gini impurity reduction across 120 decorrelated ensemble trees
            </p>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={top_features?.slice(0, 10)}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 90, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                <XAxis type="number" tick={{ fill: '#64748B', fontSize: 11 }} />
                <YAxis dataKey="feature" type="category" tick={{ fill: '#64748B', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0F172A', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                  formatter={(val) => [(val * 100).toFixed(2) + '%', 'Relative Importance']}
                />
                <Bar dataKey="importance" name="Feature Weight" fill="#0284C7" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
