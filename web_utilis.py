import requests
import logging
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class SearchResult:
    title: str
    snippet: str
    source: str = ""
    date: str = ""
    url: str = ""
    type: str = "general"
    metadata: Dict[str, Any] = None

class WebAccessManager:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.logger = logging.getLogger('WebAccessManager')
        
        # Configure default search parameters
        self.default_params = {
            "engine": "google",
            "gl": "us",
            "hl": "en"
        }
        
        # Define search type handlers
        self.search_handlers = {
            "news": self._handle_news_search,
            "weather": self._handle_weather_search,
            "sports": self._handle_sports_search,
            "scores": self._handle_sports_search,  # Alias for sports
            "price": self._handle_price_search,
            "product": self._handle_price_search,  # Alias for price
            "definition": self._handle_definition_search,
            "facts": self._handle_facts_search,
            "general": self._handle_general_search
        }

    async def process_web_query(self, query: str, search_type: str = "general") -> str:
        """Process web queries based on their type and return formatted information"""
        try:
            # Clean up search_type to match our handlers
            search_type = search_type.lower().strip()
            if search_type not in self.search_handlers:
                search_type = "general"
                
            self.logger.info(f"Processing web query: '{query}' (type: {search_type})")
            
            # Extract location and refine query
            query, location = self._extract_location_from_query(query)
            
            # Perform the search
            results = await self.search_web(query, search_type, location)
            
            if not results or (
                not results.get('top_stories', []) and 
                not results.get('organic_results', []) and
                not results.get('answer_box', {})
            ):
                return f"I'm sorry, I couldn't find any information about '{query}'."
            
            # Process the results using the appropriate handler
            handler = self.search_handlers[search_type]
            response = await handler(query, results, location)
            
            if not response.strip():
                return f"I found some information about {query}, but couldn't extract the specific details you might be looking for. Could you try asking in a different way?"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in process_web_query: {str(e)}", exc_info=True)
            return f"I encountered an error while searching for information about '{query}'. Please try again or rephrase your question."

    async def search_web(self, query: str, search_type: str = "general", location: str = "") -> Optional[Dict[str, Any]]:
        """Perform a web search using SearchAPI.io with type-specific optimizations"""
        try:
            url = "https://www.searchapi.io/api/v1/search"
            
            # Determine if we need specialized searches based on type
            if search_type == "weather":
                modified_query = self._build_weather_query(query, location)
            elif search_type == "news":
                return await self._multi_variant_search(query, location, search_type)
            elif search_type == "sports" or search_type == "scores":
                modified_query = self._build_sports_query(query, location)
            elif search_type == "price" or search_type == "product":
                modified_query = self._build_price_query(query)
            elif search_type == "definition":
                modified_query = f"define {query} meaning dictionary"
            elif search_type == "facts":
                modified_query = f"{query} facts information"
            else:
                # General search - keep the query mostly as is but ensure it's clear
                modified_query = query
            
            # Add parameters based on search type
            params = {
                "api_key": self.session.params["api_key"],
                "q": modified_query,
                "num": 10,
                **self.default_params
            }
            
            # Add location data if available and relevant
            if location and search_type in ["weather", "news", "sports"]:
                params["location"] = location
            
            # Execute the search
            self.logger.info(f"Executing search query: '{modified_query}'")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()
            
            return results

        except Exception as e:
            self.logger.error(f"Search API error: {str(e)}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, 'response'):
                self.logger.error(f"Request error details: {e.response.text if e.response else 'No response'}")
            return None

    async def _multi_variant_search(self, query: str, location: str, search_type: str) -> Dict[str, Any]:
        """Perform multiple searches with variants to get diverse results"""
        search_variants = []
        
        if search_type == "news":
            if location:
                search_variants = [
                    f"breaking news {location} today",
                    f"latest news headlines {location} today",
                    f"top news stories {location} now"
                ]
            else:
                search_variants = [
                    "breaking news worldwide today",
                    "latest news headlines today",
                    "top news stories now"
                ]
        
        # Perform multiple searches and combine results
        all_results = {'top_stories': [], 'organic_results': []}
        
        for variant in search_variants:
            params = {
                "api_key": self.session.params["api_key"],
                "q": variant,
                "engine": "google",
                "num": 5,
                **self.default_params
            }
            
            try:
                response = self.session.get("https://www.searchapi.io/api/v1/search", params=params, timeout=10)
                response.raise_for_status()
                variant_results = response.json()
                
                all_results['top_stories'].extend(variant_results.get('top_stories', []))
                all_results['organic_results'].extend(variant_results.get('organic_results', []))
                
                # Add any answer boxes if they exist
                if 'answer_box' in variant_results and variant_results['answer_box']:
                    if 'answer_box' not in all_results:
                        all_results['answer_box'] = variant_results['answer_box']
                        
            except Exception as e:
                self.logger.warning(f"Error in variant search '{variant}': {str(e)}")
                continue
                
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        
        for result in all_results['organic_results']:
            url = result.get('link', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        all_results['organic_results'] = unique_results
        return all_results

    def _extract_location_from_query(self, query: str) -> Tuple[str, str]:
        """Extract location information from the query"""
        location = ""
        clean_query = query.lower()
        
        # Common location phrases
        location_phrases = [" in ", " for ", " at ", " near ", " around "]
        
        for phrase in location_phrases:
            if phrase in clean_query:
                parts = clean_query.split(phrase, 1)
                if len(parts) > 1 and parts[1].strip():
                    # Handle cases where there might be additional qualifiers after location
                    location_part = parts[1].strip()
                    end_markers = [" today", " now", " this week", " this month", " tomorrow"]
                    
                    for marker in end_markers:
                        if marker in location_part:
                            location_part = location_part.split(marker)[0].strip()
                    
                    location = location_part
                    # Remove the location part from the query for cleaner results
                    query = parts[0].strip()
                    break
        
        return query, location

    # Query construction methods
    def _build_weather_query(self, query: str, location: str) -> str:
        """Build an optimized weather query"""
        weather_terms = ["weather", "temperature", "forecast", "rain", "snow", "humidity"]
        has_weather_term = any(term in query.lower() for term in weather_terms)
        
        if location:
            if has_weather_term:
                return f"weather forecast {location} conditions temperature today"
            else:
                return f"{query} weather forecast {location} conditions temperature today"
        else:
            if has_weather_term:
                return f"current weather forecast conditions temperature today"
            else:
                return f"{query} weather forecast conditions temperature today"

    def _build_sports_query(self, query: str, location: str) -> str:
        """Build an optimized sports query"""
        sports_terms = ["score", "game", "match", "tournament", "championship", "standings"]
        time_terms = ["latest", "recent", "today", "yesterday"]
        
        has_sports_term = any(term in query.lower() for term in sports_terms)
        has_time_term = any(term in query.lower() for term in time_terms)
        
        # Check for specific sports
        sports = ["football", "soccer", "basketball", "baseball", "hockey", "tennis", "golf", "cricket"]
        mentioned_sports = [sport for sport in sports if sport in query.lower()]
        
        if location and mentioned_sports:
            return f"{' '.join(mentioned_sports)} {location} {query} scores results"
        elif mentioned_sports:
            return f"{' '.join(mentioned_sports)} {query} scores results latest"
        elif location:
            return f"{query} sports scores results {location} {'latest' if not has_time_term else ''}"
        else:
            return f"{query} {'sports' if not has_sports_term else ''} scores results {'latest' if not has_time_term else ''}"

    def _build_price_query(self, query: str) -> str:
        """Build an optimized price/product query"""
        price_terms = ["price", "cost", "how much", "cheap", "expensive"]
        shopping_terms = ["buy", "purchase", "shop", "sale", "discount"]
        
        has_price_term = any(term in query.lower() for term in price_terms)
        has_shopping_term = any(term in query.lower() for term in shopping_terms)
        
        if has_price_term:
            return f"{query} current price cost compare"
        elif has_shopping_term:
            return f"{query} price cost compare best deals"
        else:
            return f"{query} price cost where to buy compare"

    # Result handling methods
    async def _handle_news_search(self, query: str, results: Dict[str, Any], location: str) -> str:
        """Format news search results"""
        all_stories = []
        
        # Add top stories
        if results.get('top_stories'):
            for story in results['top_stories']:
                if not story.get('snippet'): 
                    continue
                title = story.get('title', '').split('|')[0].split('-')[0].strip()
                snippet = story.get('snippet', '').strip()
                source = story.get('source', '')
                date = story.get('date', '')
                if title and snippet:
                    all_stories.append(SearchResult(
                        title=title,
                        snippet=snippet,
                        source=source,
                        date=date,
                        type="news",
                        url=story.get('link', ''),
                        metadata={"is_top": True}
                    ))

        # Add organic results
        for result in results.get('organic_results', []):
            if not result.get('snippet'): 
                continue
            title = result.get('title', '').split('|')[0].split('-')[0].strip()
            snippet = result.get('snippet', '').strip()
            source = result.get('source', '')
            date = result.get('date', '')
            if title and snippet:
                all_stories.append(SearchResult(
                    title=title,
                    snippet=snippet,
                    source=source,
                    date=date,
                    type="news",
                    url=result.get('link', ''),
                    metadata={"is_top": False}
                ))

        # Remove duplicates by title similarity
        unique_stories = []
        seen_titles = set()
        
        for story in all_stories:
            # Normalize the title to prevent near-duplicate matches
            normalized_title = re.sub(r'[^\w\s]', '', story.title.lower())
            key_words = set(normalized_title.split())
            
            # Check if we've seen a very similar title already
            if not any(len(key_words.intersection(set(re.sub(r'[^\w\s]', '', t.lower()).split()))) > len(key_words) * 0.7 for t in seen_titles):
                seen_titles.add(story.title)
                unique_stories.append(story)

        # Sort by relevance (top stories first, then by recency)
        def sort_key(story):
            date = story.date.lower() if story.date else ""
            is_top = story.metadata.get("is_top", False) if story.metadata else False
            
            if is_top:
                return (0, date)
            if 'minute' in date or 'hour' in date:
                return (1, date)
            if 'today' in date:
                return (2, date)
            return (3, date)

        unique_stories.sort(key=sort_key)

        # Format the response
        response_lines = []
        if location:
            response_lines.append(f"Latest News for {location.title()}:")
        else:
            response_lines.append("Latest News:")
            
        for story in unique_stories[:5]:  # Limit to top 5 stories
            response_lines.append(f"• {story.title}")
            response_lines.append(f"  {story.snippet}")
            if story.source or story.date:
                source_info = []
                if story.source:
                    source_info.append(story.source)
                if story.date:
                    source_info.append(story.date)
                response_lines.append(f"  Source: {', '.join(source_info)}")
            response_lines.append("")

        return "\n".join(response_lines)

    async def _handle_weather_search(self, query: str, results: Dict[str, Any], location: str) -> str:
        """Format weather search results"""
        response_lines = []
        
        # Try to find the answer box first - most likely to have weather info
        if results.get('answer_box'):
            answer_box = results['answer_box']
            if 'answer' in answer_box:
                response_lines.append(answer_box['answer'])
            elif 'snippet' in answer_box:
                response_lines.append(answer_box['snippet'])
        
        # Check for relevant organic results
        if not response_lines and results.get('organic_results'):
            # Look for weather-specific content in titles and snippets
            weather_terms = ["weather", "temperature", "forecast", "degrees", "°", "rain", "sunny", "cloudy"]
            
            for result in results['organic_results']:
                snippet = result.get('snippet', '')
                title = result.get('title', '')
                
                # Check if this result is likely about weather
                if any(term in snippet.lower() for term in weather_terms) or any(term in title.lower() for term in weather_terms):
                    # Extract temperature if present (common formats: 72°F, 22°C)
                    temp_match = re.search(r'(\d+)°([CF])', snippet + title)
                    conditions_phrases = ["conditions:", "weather:", "forecast:"]
                    
                    if temp_match:
                        temp = temp_match.group(0)
                        if location:
                            response_lines.append(f"Weather for {location.title()}: {temp}")
                        else:
                            response_lines.append(f"Current Weather: {temp}")
                    
                    # Find weather description
                    for phrase in conditions_phrases:
                        if phrase in snippet.lower():
                            conditions = snippet.lower().split(phrase)[1].strip().capitalize()
                            response_lines.append(conditions)
                            break
                    
                    # If we found useful weather info, add the snippet
                    if response_lines:
                        response_lines.append(snippet.strip())
                        break
            
            # If still no specific weather info, use the first relevant result
            if not response_lines:
                for result in results['organic_results']:
                    if any(term in result.get('snippet', '').lower() for term in weather_terms):
                        if location:
                            response_lines.append(f"Weather information for {location.title()}:")
                        else:
                            response_lines.append("Current weather information:")
                        response_lines.append(result.get('snippet', '').strip())
                        break
        
        # If still no results, provide a fallback response
        if not response_lines:
            if location:
                response_lines.append(f"I couldn't find specific weather information for {location.title()}.")
            else:
                response_lines.append("I couldn't find specific weather information for your location.")
            response_lines.append("You might want to try asking about a specific city or region.")
        
        return "\n".join(response_lines)

    async def _handle_sports_search(self, query: str, results: Dict[str, Any], location: str) -> str:
        """Format sports search results"""
        response_lines = []
        
        # Look for scores or sports results in the answer box first
        if results.get('answer_box'):
            answer_box = results['answer_box']
            if 'title' in answer_box and any(x in answer_box['title'].lower() for x in ['score', 'result', 'match']):
                response_lines.append(answer_box.get('title', ''))
                if 'answer' in answer_box:
                    response_lines.append(answer_box['answer'])
                elif 'snippet' in answer_box:
                    response_lines.append(answer_box['snippet'])
                return "\n".join(response_lines)
        
        # Look for sports tables in the search results
        if results.get('sports_results'):
            sports_data = results['sports_results']
            response_lines.append(f"Sports Results:")
            
            # Handle different formats of sports_results based on what's returned
            if isinstance(sports_data, dict):
                if 'title' in sports_data:
                    response_lines.append(sports_data['title'])
                if 'games' in sports_data:
                    for game in sports_data['games']:
                        if 'teams' in game:
                            teams = game['teams']
                            scores = game.get('scores', ['', ''])
                            response_lines.append(f"{teams[0]} {scores[0]} - {scores[1]} {teams[1]}")
            elif isinstance(sports_data, list):
                for item in sports_data:
                    if isinstance(item, dict) and 'title' in item:
                        response_lines.append(item['title'])
        
        # If no structured sports data, look for relevant organic results
        if not response_lines and results.get('organic_results'):
            sports_terms = ["score", "vs", "beat", "won", "lost", "game", "match", "final"]
            
            for result in results['organic_results']:
                snippet = result.get('snippet', '')
                title = result.get('title', '')
                
                # Check if this is likely a sports result
                if any(term in snippet.lower() for term in sports_terms) or any(term in title.lower() for term in sports_terms):
                    # Extract teams and score pattern (Team A 3 - 2 Team B)
                    score_match = re.search(r'(\w+(\s\w+){0,3})\s(\d+)\s*-\s*(\d+)\s(\w+(\s\w+){0,3})', snippet + title)
                    
                    if score_match:
                        response_lines.append(f"Match Result: {score_match.group(0)}")
                    else:
                        # Fallback to presenting the snippet
                        if "vs" in title or "vs" in snippet:
                            response_lines.append(title)
                        response_lines.append(snippet.strip())
                    break
            
            # If still nothing, use the first relevant result
            if not response_lines:
                for result in results['organic_results']:
                    if any(term in result.get('title', '').lower() for term in sports_terms):
                        response_lines.append(result.get('title', ''))
                        response_lines.append(result.get('snippet', '').strip())
                        break
        
        # Fallback response
        if not response_lines:
            sport_types = ["football", "soccer", "basketball", "baseball", "hockey", "tennis"]
            mentioned_sport = next((sport for sport in sport_types if sport in query.lower()), "sports")
            
            if location:
                response_lines.append(f"I couldn't find specific {mentioned_sport} information for {location.title()}.")
            else:
                response_lines.append(f"I couldn't find specific {mentioned_sport} information or scores.")
        
        return "\n".join(response_lines)

    async def _handle_price_search(self, query: str, results: Dict[str, Any], location: str) -> str:
        """Format price search results"""
        response_lines = []
        
        # Try to find price information in the answer box first
        if results.get('answer_box'):
            answer_box = results['answer_box']
            
            # Check if the answer box has price information
            if 'title' in answer_box and any(x in answer_box['title'].lower() for x in ['price', 'cost', '$', '£', '€']):
                response_lines.append(f"Price Information for {query}:")
                if 'answer' in answer_box:
                    response_lines.append(answer_box['answer'])
                elif 'snippet' in answer_box:
                    response_lines.append(answer_box['snippet'])
                return "\n".join(response_lines)
        
        # Look for shopping results
        if results.get('shopping_results'):
            shopping_results = results['shopping_results']
            response_lines.append(f"Price Information for {query}:")
            
            for i, item in enumerate(shopping_results[:5], 1):
                name = item.get('title', '')
                price = item.get('price', '')
                source = item.get('source', '')
                
                if name and price:
                    response_lines.append(f"{i}. {name} - {price}" + (f" from {source}" if source else ""))
            
            return "\n".join(response_lines)
        
        # Check organic results for price information
        if not response_lines and results.get('organic_results'):
            price_patterns = [
                r'\$\d+(\.\d{2})?',  # $XX.XX format
                r'£\d+(\.\d{2})?',   # £XX.XX format
                r'€\d+(\.\d{2})?',   # €XX.XX format
                r'\d+\s?(dollars|USD|GBP|EUR|pounds|euros)'  # XX dollars/USD/etc format
            ]
            
            # Look for results with price patterns
            price_results = []
            
            for result in results['organic_results']:
                snippet = result.get('snippet', '')
                title = result.get('title', '')
                full_text = title + " " + snippet
                
                # Check for price patterns
                for pattern in price_patterns:
                    matches = re.findall(pattern, full_text)
                    if matches:
                        price_results.append((result, full_text))
                        break
            
            if price_results:
                response_lines.append(f"Price Information for {query}:")
                
                for result, text in price_results[:3]:
                    # Try to extract a concise price statement
                    price_statement = None
                    
                    # Look for complete sentences containing price info
                    sentences = re.split(r'[.!?]', text)
                    for sentence in sentences:
                        for pattern in price_patterns:
                            if re.search(pattern, sentence):
                                price_statement = sentence.strip()
                                break
                        if price_statement:
                            break
                    
                    # If no sentence found, just use the snippet
                    if not price_statement:
                        price_statement = result.get('snippet', '')
                    
                    response_lines.append(f"• {price_statement}")
                    
                    # Add source if available
                    if 'source' in result:
                        response_lines.append(f"  Source: {result['source']}")
                    
                    response_lines.append("")
            
        # Fallback response
        if not response_lines:
            response_lines.append(f"I couldn't find specific pricing information for '{query}'.")
            response_lines.append("Try providing more specific product details or checking a shopping website directly.")
        
        return "\n".join(response_lines)

    async def _handle_definition_search(self, query: str, results: Dict[str, Any], location: str) -> str:
        """Format definition search results"""
        response_lines = []
        
        # Check for a dictionary definition in the answer box
        if results.get('answer_box'):
            answer_box = results['answer_box']
            
            if 'title' in answer_box and ('definition' in answer_box['title'].lower() or 'meaning' in answer_box['title'].lower()):
                if 'title' in answer_box:
                    response_lines.append(answer_box['title'])
                
                if 'answer' in answer_box:
                    response_lines.append(answer_box['answer'])
                elif 'snippet' in answer_box:
                    response_lines.append(answer_box['snippet'])
                
                # Look for specific definition formats
                if 'list' in answer_box:
                    for item in answer_box['list']:
                        response_lines.append(f"• {item}")
                
                return "\n".join(response_lines)
        
        # Check organic results for definition content
        if not response_lines and results.get('organic_results'):
            definition_sources = ['dictionary', 'merriam-webster', 'oxford', 'cambridge', 'vocabulary', 'thesaurus']
            
            # First try to find results from dictionary sites
            for result in results['organic_results']:
                source = result.get('source', '').lower()
                if any(src in source for src in definition_sources):
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    
                    response_lines.append(f"Definition of '{query}':")
                    if 'meaning' in title.lower() or 'definition' in title.lower():
                        response_lines.append(title)
                    response_lines.append(snippet)
                    response_lines.append(f"Source: {result.get('source', '')}")
                    break
            
            # If no dictionary sites found, look for definition-like content
            if not response_lines:
                for result in results['organic_results']:
                    snippet = result.get('snippet', '')
                    title = result.get('title', '')
                    
                    if ('defined as' in snippet.lower() or 'refers to' in snippet.lower() or 
                        'meaning' in title.lower() or 'definition' in title.lower()):
                        response_lines.append(f"Definition of '{query}':")
                        response_lines.append(snippet)
                        if 'source' in result:
                            response_lines.append(f"Source: {result['source']}")
                        break
        
        # Fallback response
        if not response_lines:
            response_lines.append(f"I couldn't find a specific definition for '{query}'.")
        
        return "\n".join(response_lines)

    async def _handle_facts_search(self, query: str, results: Dict[str, Any], location: str) -> str:
        """Format facts search results"""
        response_lines = []
        
        # Check for knowledge panels or answer boxes first
        if results.get('knowledge_graph'):
            kg = results['knowledge_graph']
            if 'title' in kg:
                response_lines.append(f"Information about {kg['title']}:")
                
                # Add description if available
                if 'description' in kg:
                    response_lines.append(kg['description'])
                
                # Add attributes
                if 'attributes' in kg:
                    for key, value in kg['attributes'].items():
                        response_lines.append(f"• {key}: {value}")
                
                return "\n".join(response_lines)
        
        if results.get('answer_box') and not response_lines:
            answer_box = results['answer_box']
            
            if 'title' in answer_box:
                response_lines.append(answer_box['title'])
            
            if 'answer' in answer_box:
                response_lines.append(answer_box['answer'])
            elif 'snippet' in answer_box:
                response_lines.append(answer_box['snippet'])
            
            # Add list items if available
            if 'list' in answer_box:
                for item in answer_box['list']:
                    response_lines.append(f"• {item}")
            
            if response_lines:
                return "\n".join(response_lines)
        
        # Look for factual information in organic results
        if not response_lines and results.get('organic_results'):
            factual_sources = ['wikipedia', '.edu', '.gov', 'encyclopedia', 'britannica', 'national', 'scientific']
            
            # First check for reliable sources
            for result in results['organic_results']:
                source = result.get('source', '').lower()
                url = result.get('link', '').lower()
                
                is_reliable = any(src in source for src in factual_sources) or any(src in url for src in factual_sources)
                
                if is_reliable:
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    
                    response_lines.append(f"Information about {query}:")
                    response_lines.append(snippet)
                    response_lines.append(f"Source: {source}")
                    break
            
            # If no reliable sources, look for fact-like content
            if not response_lines:
                fact_terms = ["facts about", "information on", "is a", "are the", "was established", "known for"]
                
                for result in results['organic_results']:
                    snippet = result.get('snippet', '').lower()
                    title = result.get('title', '').lower()
                    
                    if any(term in snippet for term in fact_terms) or any(term in title for term in fact_terms):
                        response_lines.append(f"Information about {query}:")
                        response_lines.append(result.get('snippet', ''))
                        if result.get('source'):
                            response_lines.append(f"Source: {result.get('source')}")
                        break
        
        # Fallback response
        if not response_lines:
            # Try to use the best organic result if nothing else worked
            if results.get('organic_results'):
                result = results['organic_results'][0]
                response_lines.append(f"Here's some information about {query}:")
                response_lines.append(result.get('snippet', ''))
                if result.get('source'):
                    response_lines.append(f"Source: {result.get('source')}")
            else:
                response_lines.append(f"I couldn't find specific facts about {query}.")
                response_lines.append("Try rephrasing your question or asking about a more specific aspect.")
        
        return "\n".join(response_lines)

    async def _handle_general_search(self, query: str, results: Dict[str, Any], location: str) -> str:
        """Format general search results with intelligent categorization"""
        response_lines = []
        
        # First, check if there's an answer box for direct answers
        if results.get('answer_box'):
            answer_box = results['answer_box']
            
            # For direct questions, the answer box is usually best
            if 'title' in answer_box:
                response_lines.append(answer_box['title'])
            
            if 'answer' in answer_box:
                response_lines.append(answer_box['answer'])
            elif 'snippet' in answer_box:
                response_lines.append(answer_box['snippet'])
            
            # Add list items if available (common for steps or procedures)
            if 'list' in answer_box:
                for item in answer_box['list']:
                    response_lines.append(f"• {item}")
                    
            # If we have a good answer box response, return it
            if len(response_lines) > 1 or (len(response_lines) == 1 and len(response_lines[0]) > 100):
                return "\n".join(response_lines)
        
        # Check for knowledge graph information (entities)
        if not response_lines and results.get('knowledge_graph'):
            kg = results['knowledge_graph']
            if 'title' in kg:
                response_lines.append(f"Information about {kg['title']}:")
                
                if 'description' in kg:
                    response_lines.append(kg['description'])
                
                if 'attributes' in kg and kg['attributes']:
                    # Show only key attributes (max 3) to keep response focused
                    relevant_attrs = 0
                    for key, value in kg['attributes'].items():
                        response_lines.append(f"• {key}: {value}")
                        relevant_attrs += 1
                        if relevant_attrs >= 3:
                            break
                
                # If we have substantial knowledge graph info, return it
                if len(response_lines) > 1:
                    return "\n".join(response_lines)
        
        # Analyze and categorize organic search results
        if results.get('organic_results'):
            # Analyze the type of results to format properly
            question_words = ["what", "how", "why", "when", "where", "who", "is", "are", "can", "do"]
            is_question = any(query.lower().startswith(word) for word in question_words) or query.endswith("?")
            
            # Check for specific content types in the results
            how_to_patterns = ["how to", "steps to", "guide for", "tutorial"]
            comparison_patterns = ["vs", "versus", "compared to", "difference between"]
            list_patterns = ["best", "top", "reasons why", "ways to"]
            
            content_type = None
            if any(pattern in query.lower() for pattern in how_to_patterns):
                content_type = "how_to"
            elif any(pattern in query.lower() for pattern in comparison_patterns):
                content_type = "comparison"
            elif any(pattern in query.lower() for pattern in list_patterns):
                content_type = "list"
            elif is_question:
                content_type = "factual"
            
            # Format response based on content type
            if content_type == "how_to":
                # For how-to queries, find step-based content
                how_to_found = False
                
                for result in results['organic_results']:
                    snippet = result.get('snippet', '')
                    title = result.get('title', '')
                    
                    # Look for content that might contain steps
                    if any(pattern in title.lower() for pattern in how_to_patterns):
                        response_lines.append(f"Here's how to {query.replace('how to', '').strip()}:")
                        response_lines.append(snippet)
                        if result.get('source'):
                            response_lines.append(f"Source: {result.get('source')}")
                        how_to_found = True
                        break
                
                if not how_to_found:
                    # Fallback to best organic result
                    result = results['organic_results'][0]
                    response_lines.append(f"Here's information on how to {query.replace('how to', '').strip()}:")
                    response_lines.append(result.get('snippet', ''))
                    if result.get('source'):
                        response_lines.append(f"Source: {result.get('source')}")
            
            elif content_type == "comparison":
                # For comparison queries, find comparison content
                for result in results['organic_results']:
                    snippet = result.get('snippet', '')
                    title = result.get('title', '')
                    
                    # Look for comparison content
                    if any(pattern in title.lower() for pattern in comparison_patterns):
                        response_lines.append(title)
                        response_lines.append(snippet)
                        if result.get('source'):
                            response_lines.append(f"Source: {result.get('source')}")
                        break
                
                # If no specific comparison found, use first result
                if not response_lines:
                    result = results['organic_results'][0] 
                    response_lines.append(result.get('title', ''))
                    response_lines.append(result.get('snippet', ''))
                    if result.get('source'):
                        response_lines.append(f"Source: {result.get('source')}")
            
            elif content_type == "list":
                # For list-type queries, try to find numbered or bullet content
                for result in results['organic_results']:
                    snippet = result.get('snippet', '')
                    title = result.get('title', '')
                    
                    # List content often has numbers (3 best, top 5, etc.)
                    if any(pattern in title.lower() for pattern in list_patterns) or re.search(r'\d+\s+best|\d+\s+top', title.lower()):
                        response_lines.append(title)
                        response_lines.append(snippet)
                        if result.get('source'):
                            response_lines.append(f"Source: {result.get('source')}")
                        break
                
                # If no specific list found, use most relevant result
                if not response_lines:
                    result = results['organic_results'][0]
                    response_lines.append(result.get('title', ''))
                    response_lines.append(result.get('snippet', ''))
                    if result.get('source'):
                        response_lines.append(f"Source: {result.get('source')}")
            
            else:  # factual or other
                # Try to find the most informative result
                best_result = None
                longest_snippet = 0
                
                for result in results['organic_results'][:3]:  # Check top 3 results
                    snippet = result.get('snippet', '')
                    
                    # Prefer longer snippets as they often contain more info
                    if len(snippet) > longest_snippet:
                        longest_snippet = len(snippet)
                        best_result = result
                
                if best_result:
                    response_lines.append(f"Here's information about {query}:")
                    response_lines.append(best_result.get('snippet', ''))
                    if best_result.get('source'):
                        response_lines.append(f"Source: {best_result.get('source')}")
        
        # Fallback response if no suitable content was found
        if not response_lines:
            response_lines.append(f"I found information about {query}, but couldn't determine the most relevant details.")
            response_lines.append("Try asking a more specific question about this topic.")
        
        return "\n".join(response_lines)