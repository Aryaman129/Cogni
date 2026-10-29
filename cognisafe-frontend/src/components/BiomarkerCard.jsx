import React from "react";

/**
 * BiomarkerCard: Renders one of the 14 canonical speech biomarkers.
 * Supports:
 * - Direct scalar values with units and trends
 * - TypedFeatureResult objects with status ("valid", "insufficient_input", "extraction_failed", etc.)
 * - Informative tooltips and badges for non-valid statuses
 * - Accessible, responsive UI adhering to the Navy/Gold brand palette
 */
const BiomarkerCard = ({
  name,
  value,
  status = "good",
  trend,
  typedResult = null,
  unit = "",
  tooltip = "",
  isProxy = false,
}) => {
  // Extract metadata if typedResult is provided
  const effectiveStatus = typedResult?.status || status;
  const effectiveValue = typedResult?.value !== undefined ? typedResult.value : value;
  const effectiveUnit = typedResult?.unit || unit;
  const effectiveReason = typedResult?.reason || tooltip;
  const effectiveIsProxy = typedResult?.is_proxy ?? isProxy;
  const effectiveMethod = typedResult?.method || "";

  // Check for typed non-valid statuses
  const isInsufficient = effectiveStatus === "insufficient_input";
  const isFailed = effectiveStatus === "extraction_failed";
  const isUnsupported = effectiveStatus === "unsupported_input";
  const isInvalid = effectiveStatus === "invalid_measurement";
  const isMissingDep = effectiveStatus === "missing_dependency";
  const hasTypedError = isInsufficient || isFailed || isUnsupported || isInvalid || isMissingDep;

  // Determine badge label and class
  let statusBadgeLabel = "Normal";
  let statusBadgeClass = "good";

  if (hasTypedError) {
    if (isInsufficient) {
      statusBadgeLabel = "Insufficient Input";
      statusBadgeClass = "insufficient";
    } else if (isFailed) {
      statusBadgeLabel = "Extraction Failed";
      statusBadgeClass = "failed";
    } else if (isUnsupported) {
      statusBadgeLabel = "Unsupported";
      statusBadgeClass = "unsupported";
    } else if (isMissingDep) {
      statusBadgeLabel = "Missing Dependency";
      statusBadgeClass = "warning";
    } else {
      statusBadgeLabel = "Invalid Measurement";
      statusBadgeClass = "failed";
    }
  } else if (effectiveStatus === "warning" || effectiveStatus === "warn") {
    statusBadgeLabel = "Watch";
    statusBadgeClass = "warning";
  } else if (effectiveStatus === "alert" || effectiveStatus === "danger" || effectiveStatus === "crit") {
    statusBadgeLabel = "Alert";
    statusBadgeClass = "danger";
  } else if (effectiveStatus === "baseline" || effectiveStatus === "neutral") {
    statusBadgeLabel = "Baseline";
    statusBadgeClass = "neutral";
  }

  // Value formatting
  let displayValue = "—";
  if (!hasTypedError && effectiveValue !== null && effectiveValue !== undefined && effectiveValue !== "—") {
    displayValue = typeof effectiveValue === "number"
      ? (Number.isInteger(effectiveValue) ? effectiveValue.toString() : effectiveValue.toFixed(2))
      : String(effectiveValue);
    if (effectiveUnit && !displayValue.includes(effectiveUnit)) {
      displayValue = `${displayValue} ${effectiveUnit}`.trim();
    }
  }

  return (
    <div
      className={`biomarker-card ${hasTypedError ? `card-${statusBadgeClass}` : ""}`}
      title={effectiveReason || `${name}: ${displayValue}`}
    >
      <div className="biomarker-header">
        <span className="biomarker-name">
          {name}
          {effectiveIsProxy && (
            <span
              className="proxy-indicator"
              title={effectiveMethod ? `Proxy measurement: ${effectiveMethod}` : "Spectral proxy measurement"}
            >
              [Proxy]
            </span>
          )}
        </span>
        <span
          className={`biomarker-status ${statusBadgeClass}`}
          title={statusBadgeLabel}
        />
      </div>

      <div className="biomarker-value-container">
        {hasTypedError ? (
          <div className="biomarker-typed-badge" title={effectiveReason || statusBadgeLabel}>
            <span className="typed-icon">ℹ️</span>
            <span className="typed-text">{statusBadgeLabel}</span>
          </div>
        ) : (
          <div className="biomarker-value">{displayValue}</div>
        )}
      </div>

      {hasTypedError ? (
        <div className="biomarker-reason-text" title={effectiveReason}>
          {effectiveReason || "Insufficient data to measure this feature"}
        </div>
      ) : (
        <>
          <div className={`biomarker-trend ${trend?.type || "flat"}`}>
            {trend?.arrow || "●"} {trend?.text || "Stable"}
          </div>
          <div className="biomarker-progress">
            <div
              className={`progress-fill ${statusBadgeClass}`}
              style={{ width: `${Math.min(100, Math.max(5, trend?.percentage || 50))}%` }}
            />
          </div>
        </>
      )}
    </div>
  );
};

export default BiomarkerCard;
