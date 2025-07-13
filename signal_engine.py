"""
signal_engine.py - M.A.N.T.R.A. Signal Generation Engine
======================================================
Multi-factor scoring system for stock analysis
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Optional
import logging
from constants import (
    FACTOR_WEIGHTS, SIGNAL_LEVELS, MOMENTUM_THRESHOLDS,
    VOLUME_THRESHOLDS, PE_RANGES
)

logger = logging.getLogger(__name__)

class SignalEngine:
    """Clean and efficient signal generation"""
    
    @staticmethod
    @st.cache_data(ttl=60, show_spinner=False)
    def calculate_all_signals(stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all signals and scores for stocks
        Returns enhanced dataframe with scores and decisions
        """
        if stocks_df.empty:
            return stocks_df
        
        df = stocks_df.copy()
        
        # Calculate individual factor scores
        df['momentum_score'] = SignalEngine._calculate_momentum_score(df)
        df['value_score'] = SignalEngine._calculate_value_score(df)
        df['technical_score'] = SignalEngine._calculate_technical_score(df)
        df['volume_score'] = SignalEngine._calculate_volume_score(df)
        df['fundamental_score'] = SignalEngine._calculate_fundamental_score(df)
        
        # Add sector strength if available
        if not sector_df.empty and 'sector' in df.columns:
            df = SignalEngine._add_sector_strength(df, sector_df)
        
        # Calculate composite score
        df['composite_score'] = SignalEngine._calculate_composite_score(df)
        
        # Make decisions
        df['decision'] = df['composite_score'].apply(SignalEngine._get_decision)
        
        # Calculate risk score
        df['risk_score'] = SignalEngine._calculate_risk_score(df)
        df['risk_level'] = df['risk_score'].apply(SignalEngine._get_risk_level)
        
        # Add opportunity score (upside potential)
        df['opportunity_score'] = SignalEngine._calculate_opportunity_score(df)
        
        # Add reasoning
        df['reasoning'] = df.apply(SignalEngine._generate_reasoning, axis=1)
        
        # Rank stocks
        df['rank'] = df['composite_score'].rank(ascending=False, method='min')
        df['percentile'] = (df['composite_score'].rank(pct=True) * 100).round(1)
        
        return df
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score based on returns"""
        score = pd.Series(50.0, index=df.index)
        
        # Weight recent periods more heavily
        weights = {'ret_1d': 0.1, 'ret_7d': 0.2, 'ret_30d': 0.3, 'ret_3m': 0.4}
        
        for period, weight in weights.items():
            if period in df.columns:
                returns = df[period].fillna(0)
                
                # Score based on strength
                if period == 'ret_1d':
                    threshold = MOMENTUM_THRESHOLDS['STRONG']['1d']
                elif period == 'ret_7d':
                    threshold = MOMENTUM_THRESHOLDS['STRONG']['7d']
                elif period == 'ret_30d':
                    threshold = MOMENTUM_THRESHOLDS['STRONG']['30d']
                else:
                    threshold = 10  # Default for 3m
                
                # Linear scoring with caps
                period_score = 50 + (returns / threshold) * 25
                period_score = period_score.clip(0, 100)
                
                # Add weighted contribution
                score += (period_score - 50) * weight
        
        return score.clip(0, 100).round(1)
    
    @staticmethod
    def _calculate_value_score(df: pd.DataFrame) -> pd.Series:
        """Calculate value score based on PE and growth"""
        score = pd.Series(50.0, index=df.index)
        
        # PE Ratio Score (60% weight) - lower is better
        if 'pe' in df.columns:
            pe = df['pe'].fillna(df['pe'].median())
            
            # Assign scores based on PE ranges
            conditions = [
                (pe > 0) & (pe <= PE_RANGES['UNDERVALUED'][1]),
                (pe > PE_RANGES['UNDERVALUED'][1]) & (pe <= PE_RANGES['FAIR'][1]),
                (pe > PE_RANGES['FAIR'][1]) & (pe <= PE_RANGES['OVERVALUED'][1]),
                pe > PE_RANGES['OVERVALUED'][1],
                pe <= 0
            ]
            choices = [90, 70, 50, 30, 20]
            
            pe_score = pd.Series(np.select(conditions, choices, default=50), index=df.index)
            score = score * 0.4 + pe_score * 0.6
        
        # EPS Growth Score (40% weight) - higher is better
        if 'eps_change_pct' in df.columns:
            eps_growth = df['eps_change_pct'].fillna(0)
            
            # Score based on growth
            growth_score = pd.Series(50.0, index=df.index)
            growth_score[eps_growth > 50] = 90
            growth_score[eps_growth.between(25, 50)] = 75
            growth_score[eps_growth.between(10, 25)] = 60
            growth_score[eps_growth.between(0, 10)] = 50
            growth_score[eps_growth < 0] = 30
            
            score = score * 0.6 + growth_score * 0.4
        
        return score.clip(0, 100).round(1)
    
    @staticmethod
    def _calculate_technical_score(df: pd.DataFrame) -> pd.Series:
        """Calculate technical score based on price position and trends"""
        score = pd.Series(50.0, index=df.index)
        
        # Price vs SMAs (60% weight)
        sma_score = pd.Series(50.0, index=df.index)
        sma_count = 0
        
        for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
            if sma in df.columns and 'price' in df.columns:
                above_sma = df['price'] > df[sma]
                sma_score += above_sma * 10  # +10 for each SMA above
                sma_count += 1
        
        if sma_count > 0:
            score = score * 0.4 + sma_score * 0.6
        
        # 52-week position (40% weight)
        if 'position_52w' in df.columns:
            pos_score = pd.Series(50.0, index=df.index)
            
            # Sweet spot is 30-70% of range
            pos_score[df['position_52w'].between(30, 70)] = 70
            pos_score[df['position_52w'].between(20, 30)] = 60
            pos_score[df['position_52w'].between(70, 80)] = 60
            pos_score[df['position_52w'] > 80] = 50  # Overbought
            pos_score[df['position_52w'] < 20] = 40  # Oversold
            
            score = score * 0.6 + pos_score * 0.4
        
        return score.clip(0, 100).round(1)
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate volume score based on activity"""
        score = pd.Series(50.0, index=df.index)
        
        # Relative volume (rvol)
        if 'rvol' in df.columns:
            rvol = df['rvol'].fillna(1.0)
            
            # Score based on volume levels
            conditions = [
                rvol >= VOLUME_THRESHOLDS['SPIKE'],
                rvol >= VOLUME_THRESHOLDS['HIGH'],
                rvol >= VOLUME_THRESHOLDS['NORMAL'],
                rvol >= VOLUME_THRESHOLDS['LOW'],
                rvol < VOLUME_THRESHOLDS['LOW']
            ]
            choices = [85, 70, 50, 30, 20]
            
            score = pd.Series(np.select(conditions, choices, default=50), index=df.index)
        
        # Boost score if volume spike with positive price action
        if 'rvol' in df.columns and 'ret_1d' in df.columns:
            volume_price_boost = ((df['rvol'] > 2) & (df['ret_1d'] > 1)) * 10
            score = (score + volume_price_boost).clip(0, 100)
        
        return score.round(1)
    
    @staticmethod
    def _calculate_fundamental_score(df: pd.DataFrame) -> pd.Series:
        """Calculate fundamental score based on quality metrics"""
        score = pd.Series(50.0, index=df.index)
        factors = 0
        
        # EPS trend
        if 'eps_change_pct' in df.columns:
            eps_score = pd.Series(50.0, index=df.index)
            eps_score[df['eps_change_pct'] > 20] = 70
            eps_score[df['eps_change_pct'] > 0] = 60
            eps_score[df['eps_change_pct'] < -20] = 30
            
            score += eps_score
            factors += 1
        
        # Market cap quality
        if 'market_cap' in df.columns:
            mcap_score = pd.Series(50.0, index=df.index)
            mcap_score[df['market_cap'] > 1e11] = 70  # Large caps are safer
            mcap_score[df['market_cap'].between(2e10, 1e11)] = 60
            mcap_score[df['market_cap'] < 5e9] = 40  # Small caps are riskier
            
            score += mcap_score
            factors += 1
        
        # Profitability
        if 'pe' in df.columns:
            profit_score = pd.Series(50.0, index=df.index)
            profit_score[(df['pe'] > 0) & (df['pe'] < 30)] = 70  # Profitable with reasonable PE
            profit_score[df['pe'] <= 0] = 30  # Loss making
            
            score += profit_score
            factors += 1
        
        if factors > 0:
            score = score / (factors + 1)  # Average all factors
        
        return score.clip(0, 100).round(1)
    
    @staticmethod
    def _add_sector_strength(df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Add sector strength to stocks"""
        # Create sector performance mapping
        if 'sector_ret_30d' in sector_df.columns:
            sector_perf = sector_df.set_index('sector')['sector_ret_30d'].to_dict()
            df['sector_performance'] = df['sector'].map(sector_perf).fillna(0)
            
            # Convert to score (normalize to 0-100)
            if df['sector_performance'].std() > 0:
                df['sector_score'] = (
                    50 + (df['sector_performance'] - df['sector_performance'].mean()) / 
                    df['sector_performance'].std() * 20
                ).clip(0, 100)
            else:
                df['sector_score'] = 50.0
        else:
            df['sector_score'] = 50.0
        
        return df
    
    @staticmethod
    def _calculate_composite_score(df: pd.DataFrame) -> pd.Series:
        """Calculate weighted composite score"""
        # Get available scores
        score_columns = {
            'momentum': 'momentum_score',
            'value': 'value_score',
            'technical': 'technical_score',
            'volume': 'volume_score',
            'fundamentals': 'fundamental_score'
        }
        
        composite = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for factor, col in score_columns.items():
            if col in df.columns and factor in FACTOR_WEIGHTS:
                weight = FACTOR_WEIGHTS[factor]
                composite += df[col].fillna(50) * weight
                total_weight += weight
        
        # Normalize if weights don't sum to 1
        if total_weight > 0:
            composite = composite / total_weight
        
        return composite.round(1)
    
    @staticmethod
    def _get_decision(score: float) -> str:
        """Convert score to trading decision"""
        if score >= SIGNAL_LEVELS['BUY']:
            return 'BUY'
        elif score >= SIGNAL_LEVELS['WATCH']:
            return 'WATCH'
        elif score >= SIGNAL_LEVELS['AVOID']:
            return 'NEUTRAL'
        else:
            return 'AVOID'
    
    @staticmethod
    def _calculate_risk_score(df: pd.DataFrame) -> pd.Series:
        """Calculate risk score (0-100, higher is riskier)"""
        risk = pd.Series(0.0, index=df.index)
        factors = 0
        
        # Volatility risk (using return variance)
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            returns_std = df[['ret_1d', 'ret_7d', 'ret_30d']].std(axis=1)
            volatility_risk = (returns_std / 10 * 30).clip(0, 30)  # Max 30 points
            risk += volatility_risk
            factors += 1
        
        # Valuation risk
        if 'pe' in df.columns:
            pe_risk = pd.Series(0.0, index=df.index)
            pe_risk[df['pe'] > 40] = 20
            pe_risk[df['pe'] < 0] = 30  # Loss making is risky
            pe_risk[df['pe'].between(25, 40)] = 10
            risk += pe_risk
            factors += 1
        
        # Size risk (smaller = riskier)
        if 'market_cap' in df.columns:
            size_risk = pd.Series(10.0, index=df.index)
            size_risk[df['market_cap'] < 5e9] = 30  # Small cap
            size_risk[df['market_cap'].between(5e9, 2e10)] = 20  # Mid cap
            size_risk[df['market_cap'] > 1e11] = 0  # Large cap
            risk += size_risk
            factors += 1
        
        # Liquidity risk
        if 'volume_1d' in df.columns:
            liquidity_risk = pd.Series(0.0, index=df.index)
            liquidity_risk[df['volume_1d'] < 50000] = 20
            liquidity_risk[df['volume_1d'] < 10000] = 30
            risk += liquidity_risk
            factors += 1
        
        return risk.clip(0, 100).round(1)
    
    @staticmethod
    def _get_risk_level(risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score <= 40:
            return 'LOW'
        elif risk_score <= 70:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    @staticmethod
    def _calculate_opportunity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate opportunity score (potential upside)"""
        # Combination of high score + low risk + momentum
        score = df.get('composite_score', 50)
        risk = df.get('risk_score', 50)
        momentum = df.get('momentum_score', 50)
        
        # Formula: High score, low risk, good momentum
        opportunity = (score * 0.5 + (100 - risk) * 0.3 + momentum * 0.2)
        
        return opportunity.round(1)
    
    @staticmethod
    def _generate_reasoning(row: pd.Series) -> str:
        """Generate human-readable reasoning for decision"""
        reasons = []
        
        # Score-based reasoning
        score = row.get('composite_score', 50)
        if score >= 85:
            reasons.append("Excellent overall score")
        elif score >= 75:
            reasons.append("Strong fundamentals")
        elif score < 40:
            reasons.append("Weak indicators")
        
        # Momentum
        if row.get('momentum_score', 50) > 80:
            reasons.append("Strong momentum")
        elif row.get('momentum_score', 50) < 30:
            reasons.append("Poor momentum")
        
        # Value
        if row.get('value_score', 50) > 80:
            reasons.append("Attractive valuation")
        
        # Volume
        if row.get('volume_score', 50) > 80:
            reasons.append("High volume activity")
        
        # Risk
        risk_level = row.get('risk_level', 'MEDIUM')
        if risk_level == 'HIGH':
            reasons.append("High risk")
        elif risk_level == 'LOW':
            reasons.append("Low risk profile")
        
        # Combine top 3 reasons
        return " | ".join(reasons[:3]) if reasons else "Mixed signals"
