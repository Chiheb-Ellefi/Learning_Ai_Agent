# model/smart_resource_ranker.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SmartResourceRanker:
    """
    Ranks and recommends resources WITHOUT needing a dataset
    Uses NLP techniques: TF-IDF + Cosine Similarity
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.resource_vectors = None
        self.resources = None
    
    def add_resources(self, resources):
        """
        Add resources from the generated roadmap
        No external dataset needed - uses roadmap data!
        
        resources = [
            {
                'title': 'React Official Docs',
                'type': 'documentation',
                'description': 'Official React documentation with examples',
                'url': 'https://react.dev',
                'difficulty': 'intermediate',
                'tags': ['interactive', 'official', 'comprehensive']
            },
            ...
        ]
        """
        self.resources = resources
        
        # Create text representation of each resource
        resource_texts = [
            f"{r['title']} {r['type']} {r.get('description', '')} {' '.join(r.get('tags', []))}"
            for r in resources
        ]
        
        # Vectorize
        self.resource_vectors = self.vectorizer.fit_transform(resource_texts)
    
    def rank_for_topic(self, topic_query, learning_style, difficulty_level):
        """
        Rank resources for a specific topic and user
        
        Args:
            topic_query: "React hooks and state management"
            learning_style: "Visual" | "Auditory" | "Reading" | "Kinesthetic"
            difficulty_level: 0.0 to 1.0
        """
        
        # 1. Text similarity scoring
        query_vector = self.vectorizer.transform([topic_query])
        text_scores = cosine_similarity(query_vector, self.resource_vectors)[0]
        
        # 2. Learning style bonus
        style_scores = self._calculate_style_bonus(learning_style)
        
        # 3. Difficulty matching
        difficulty_scores = self._calculate_difficulty_match(difficulty_level)
        
        # 4. Combine scores with weights
        final_scores = (
            text_scores * 0.5 +           # 50% - relevance to topic
            style_scores * 0.3 +          # 30% - matches learning style
            difficulty_scores * 0.2       # 20% - appropriate difficulty
        )
        
        # 5. Rank resources
        ranked_indices = np.argsort(final_scores)[::-1]  # Descending order
        
        ranked_resources = []
        for idx in ranked_indices[:10]:  # Top 10
            resource = self.resources[idx].copy()
            resource['relevance_score'] = float(final_scores[idx])
            resource['match_reason'] = self._explain_match(
                text_scores[idx],
                style_scores[idx],
                difficulty_scores[idx]
            )
            ranked_resources.append(resource)
        
        return ranked_resources
    
    def _calculate_style_bonus(self, learning_style):
        """Give bonus points based on learning style preference"""
        scores = np.zeros(len(self.resources))
        
        for i, resource in enumerate(self.resources):
            resource_type = resource['type'].lower()
            
            if learning_style == 'Visual':
                if 'video' in resource_type or 'diagram' in resource_type:
                    scores[i] = 1.0
                elif 'interactive' in resource_type:
                    scores[i] = 0.8
                else:
                    scores[i] = 0.3
                    
            elif learning_style == 'Auditory':
                if 'video' in resource_type or 'podcast' in resource_type:
                    scores[i] = 1.0
                elif 'tutorial' in resource_type:
                    scores[i] = 0.7
                else:
                    scores[i] = 0.4
                    
            elif learning_style == 'Reading':
                if 'article' in resource_type or 'documentation' in resource_type:
                    scores[i] = 1.0
                elif 'book' in resource_type:
                    scores[i] = 0.9
                else:
                    scores[i] = 0.3
                    
            elif learning_style == 'Kinesthetic':
                if 'interactive' in resource_type or 'tutorial' in resource_type:
                    scores[i] = 1.0
                elif 'exercise' in resource_type:
                    scores[i] = 0.9
                else:
                    scores[i] = 0.4
        
        return scores
    
    def _calculate_difficulty_match(self, user_difficulty):
        """Match resources to user's current difficulty level"""
        scores = np.zeros(len(self.resources))
        
        difficulty_map = {
            'beginner': 0.3,
            'intermediate': 0.6,
            'advanced': 0.9
        }
        
        for i, resource in enumerate(self.resources):
            resource_difficulty = difficulty_map.get(
                resource.get('difficulty', 'intermediate'),
                0.6
            )
            
            # Score based on how close difficulty levels are
            diff = abs(resource_difficulty - user_difficulty)
            scores[i] = 1.0 - diff  # Closer = higher score
        
        return scores
    
    def _explain_match(self, text_score, style_score, difficulty_score):
        """Generate explanation for why resource was recommended"""
        reasons = []
        
        if text_score > 0.7:
            reasons.append("Highly relevant to topic")
        if style_score > 0.8:
            reasons.append("Matches your learning style")
        if difficulty_score > 0.8:
            reasons.append("Appropriate difficulty level")
        
        return " • ".join(reasons) if reasons else "Good general resource"
    
    def get_diverse_recommendations(self, topic_query, learning_style, 
                                   difficulty_level, n_recommendations=5):
        """
        Get diverse set of recommendations (mix of types)
        """
        ranked = self.rank_for_topic(topic_query, learning_style, difficulty_level)
        
        # Ensure diversity in types
        diverse_recs = []
        seen_types = set()
        
        for resource in ranked:
            resource_type = resource['type']
            
            # Add if we haven't seen this type yet, or if we have few recommendations
            if resource_type not in seen_types or len(diverse_recs) < n_recommendations:
                diverse_recs.append(resource)
                seen_types.add(resource_type)
                
                if len(diverse_recs) >= n_recommendations:
                    break
        
        return diverse_recs


# Example usage - No dataset needed!
def test_resource_ranker():
    """
    Test with resources from your generated roadmap
    """
    
    # These come from your Claude API roadmap generation
    sample_resources = [
        {
            'title': 'React Official Documentation',
            'type': 'documentation',
            'description': 'Official React docs with interactive examples',
            'url': 'https://react.dev',
            'difficulty': 'intermediate',
            'tags': ['official', 'interactive', 'comprehensive']
        },
        {
            'title': 'React Hooks Video Tutorial',
            'type': 'video',
            'description': 'Complete video course on React Hooks',
            'url': 'https://youtube.com/...',
            'difficulty': 'beginner',
            'tags': ['video', 'tutorial', 'hooks']
        },
        {
            'title': 'Advanced React Patterns',
            'type': 'article',
            'description': 'In-depth article on advanced patterns',
            'url': 'https://...',
            'difficulty': 'advanced',
            'tags': ['patterns', 'advanced']
        }
    ]
    
    ranker = SmartResourceRanker()
    ranker.add_resources(sample_resources)
    
    # Get recommendations for a visual learner studying hooks
    recommendations = ranker.rank_for_topic(
        topic_query="React hooks useState useEffect",
        learning_style="Visual",
        difficulty_level=0.4  # Beginner-intermediate
    )
    
    for rec in recommendations:
        print(f"- {rec['title']} (Score: {rec['relevance_score']:.2f})")
        print(f"  Reason: {rec['match_reason']}")